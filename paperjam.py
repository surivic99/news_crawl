import os
import json
import csv
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin, quote
from datetime import datetime, timedelta
import time
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    {
        "keywords": "BGL BNP PARIBAS",
        "base_url": "https://paperjam.lu"
    },
]

SEARCH_URL_TEMPLATE = "https://paperjam.lu/search?numericRefinementList%5BpublicationDate%5D=Tous&query={}"

OUTPUT_DIR = "scraped_data"

class News(ABC):
    def __init__(self, url: str, title: str, keyword: str, date: datetime):
        if not isinstance(date, datetime):
             raise TypeError("Date must be a datetime object")
        self.url = url
        self.title = title
        self.keyword = keyword
        self.date = date

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class PaperJamNews(News):
    def __init__(self, url: str, title: str, keyword: str, date: datetime):
        super().__init__(url, title, keyword, date)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "keyword": self.keyword,
            "date": self.date.strftime('%Y-%m-%d') if self.date else None
        }

class PaperJamCrawler:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.driver = None
        self.wait = None
        self._setup_directories()

    def _setup_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _init_driver(self):
        logger.info("Initializing WebDriver...")
        edge_options = webdriver.EdgeOptions()
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--disable-web-security")
        edge_options.add_argument("--disable-features=VizDisplayCompositor")
        edge_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        service = EdgeService(executable_path="msedgedriver.exe")
        self.driver = webdriver.Edge(service=service, options=edge_options)
        self.wait = WebDriverWait(self.driver, 10)

    def _handle_cookies(self):
        logger.debug("Handling cookie consent...")
        selectors = [
            (By.ID, "didomi-notice-agree-button"),
            (By.CSS_SELECTOR, "button.didomi-dismiss-button"),
            (By.XPATH, "//button[contains(@aria-label, 'Accepter') and contains(@aria-label, 'Fermer')]"),
            (By.XPATH, "//button[contains(text(), 'Accept all') or contains(text(), 'Accepter')]"),
            (By.XPATH, "//button[contains(text(), 'I agree') or contains(text(), 'J\'accepte')]"),
            (By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'ACCEPT')]"),
            (By.CSS_SELECTOR, "#L2AGLb"),
            (By.CSS_SELECTOR, "button[data-accept-cookies]"),
            (By.CSS_SELECTOR, ".cookie-banner .accept-button"),
            (By.CSS_SELECTOR, ".gdpr-accept-button")
        ]

        for selector_type, selector_value in selectors:
            try:
                element = self.wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                element.click()
                logger.debug(f"Clicked cookie button with selector: {selector_value}")
                time.sleep(1)
                return True
            except (TimeoutException, NoSuchElementException):
                continue
            except Exception as e:
                logger.warning(f"Error clicking cookie button {selector_value}: {e}")
                continue
        logger.debug("No cookie consent button found or clickable.")
        return False

    def _parse_date(self, date_string: str) -> datetime:
        if not date_string:
            return None
        try:
            if "•" in date_string:
                date_part = date_string.split("•")[-1].strip()
            else:
                date_part = date_string.strip()

            parts = date_part.split()
            if parts:
                date_str = parts[-1]
                return datetime.strptime(date_str, "%d.%m.%Y")
            return None
        except ValueError as e:
            logger.error(f"Error parsing date '{date_string}': {e}")
            return None

    def _filter_by_date(self, news_list: list[PaperJamNews], days_back: int) -> list[PaperJamNews]:
        if days_back < 0:
            logger.warning("Days back cannot be negative. Returning all articles.")
            return news_list

        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_news = [news for news in news_list if news.date and news.date.date() >= cutoff_date.date()]
        logger.info(f"Filtered {len(news_list)} articles. Keeping {len(filtered_news)} from the last {days_back} day(s).")
        return filtered_news

    def crawl(self, search_queries: list, days_back: int = 30) -> list[PaperJamNews]:
        logger.info("Starting PaperJam Crawler...")
        all_news = []

        self._init_driver()

        try:
            for i, query in enumerate(search_queries, 1):
                keyword = query['keywords']
                base_url = query['base_url']
                logger.info(f"Performing search #{i}: Keyword='{keyword}'")

                try:
                    search_url = SEARCH_URL_TEMPLATE.format(quote(keyword))
                    logger.debug(f"Visiting URL: {search_url}")
                    self.driver.get(search_url)
                    time.sleep(3)

                    self._handle_cookies()

                    results = self.driver.find_elements(By.CSS_SELECTOR, ".search__results-item .news-card")

                    if not results:
                        logger.info("No results found for this query.")
                        continue

                    logger.info(f"Found {len(results)} potential results. Processing...")

                    for j, result in enumerate(results):
                        try:
                            parent_item = result.find_element(By.XPATH, "./ancestor::div[contains(@class, 'search__results-item')]")
                            link_element = parent_item.find_element(By.TAG_NAME, "a")
                            relative_url = link_element.get_attribute('href')

                            if not relative_url:
                                logger.warning(f"Skipping result {j+1} - No URL found.")
                                continue

                            url = urljoin(base_url, relative_url)

                            title_element = result.find_element(By.CSS_SELECTOR, "h4.news-card__title")
                            title = title_element.text.strip()

                            date_element = result.find_element(By.CSS_SELECTOR, ".informations")
                            date_text = date_element.text
                            publication_date = self._parse_date(date_text)

                            if not publication_date:
                                logger.warning(f"Could not parse date for article '{title}'. Skipping.")
                                continue

                            news_item = PaperJamNews(url=url, title=title, keyword=keyword, date=publication_date)
                            all_news.append(news_item)
                            logger.debug(f"Added article: {title} ({publication_date.strftime('%Y-%m-%d')})")

                        except NoSuchElementException as e:
                            logger.warning(f"Missing element in result {j+1} for query '{keyword}': {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing result {j+1} for query '{keyword}': {e}")
                            continue

                except Exception as e:
                    logger.error(f"An error occurred during search #{i} for '{keyword}': {e}")
                    continue

            logger.info(f"Applying date filter: last {days_back} day(s).")
            filtered_news = self._filter_by_date(all_news, days_back)

        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver closed.")

        return filtered_news

    def save_to_json(self, news_list: list[PaperJamNews], filename: str):
        filepath = os.path.join(self.output_dir, filename)
        data_to_save = [news.to_dict() for news in news_list]

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data_to_save)} articles to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data to JSON: {e}")

    def save_to_csv(self, news_list: list[PaperJamNews], filename: str):
        filepath = os.path.join(self.output_dir, filename)
        
        if not news_list:
            logger.warning("No data to save to CSV.")
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["url", "title", "keyword", "date"])
            return

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["url", "title", "keyword", "date"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for news in news_list:
                    writer.writerow(news.to_dict())
            logger.info(f"Saved {len(news_list)} articles to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data to CSV: {e}")


if __name__ == "__main__":
    DAYS_BACK = 30
    JSON_FILENAME = "scraped_news.json"
    CSV_FILENAME = "scraped_news.csv"

    crawler = PaperJamCrawler()

    try:
        scraped_news = crawler.crawl(search_queries=SEARCH_QUERIES, days_back=DAYS_BACK)
        crawler.save_to_json(scraped_news, JSON_FILENAME)
        crawler.save_to_csv(scraped_news, CSV_FILENAME)
        logger.info("Scraping process completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred in the main execution: {e}")
