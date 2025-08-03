import time
import os
import requests
import json
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from urllib.parse import urljoin
import re
from datetime import datetime

SEARCH_QUERIES = [
    {
        "keywords": "BGL BNP PARIBAS",
        "base_url": "https://www.luxtimes.lu"
    },
]

SEARCH_URL_TEMPLATE = "https://www.luxtimes.lu/search/?q={}"

OUTPUT_DIR = "scraped_pages"
MAX_PAGES_TO_SAVE = 5

def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def parse_date(date_string):
    """Parse various date formats from LuxTimes."""
    if not date_string:
        return None
    
    date_string = date_string.strip()
    
    # Try different date formats
    formats = [
        "%d/%m/%Y",      # 20/07/2025
        "%d.%m.%Y",      # 27.06.2019
        "%Y-%m-%d",      # 2019-06-27
        "%B %d, %Y",     # June 27, 2019
        "%b %d, %Y",     # Jun 27, 2019
        "%d %B %Y",      # 27 June 2019
        "%d %b %Y",      # 27 Jun 2019
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    # If specific formats fail, try more flexible parsing
    try:
        # Handle dates like "Published on 27.06.2019"
        if "Published" in date_string:
            date_part = date_string.split("Published")[-1].strip()
            return parse_date(date_part)
        # Handle other common patterns
        elif "on" in date_string:
            date_part = date_string.split("on")[-1].strip()
            return parse_date(date_part)
    except:
        pass
    
    return None

def save_html_page(url, title, search_index, result_index):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        safe_title = sanitize_filename(title)
        filename = f"search_{search_index}_result_{result_index}_{safe_title}.html"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(response.text)
        return True
    except Exception:
        return False

def handle_cookies(driver, wait):
    didomi_selectors = [
        (By.ID, "didomi-notice-agree-button"),
        (By.CSS_SELECTOR, "button.didomi-dismiss-button"),
        (By.XPATH, "//button[contains(@aria-label, 'Accepter') and contains(@aria-label, 'Reject')]")
    ]
    
    general_selectors = [
        (By.XPATH, "//button[contains(text(), 'Accept all') or contains(text(), 'Accepter')]"),
        (By.XPATH, "//button[contains(text(), 'I agree') or contains(text(), 'J\'accepte')]"),
        (By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'ACCEPT')]"),
        (By.CSS_SELECTOR, "#L2AGLb"),
        (By.CSS_SELECTOR, "button[data-accept-cookies]"),
        (By.CSS_SELECTOR, ".cookie-banner .accept-button"),
        (By.CSS_SELECTOR, ".gdpr-accept-button")
    ]
    
    all_selectors = didomi_selectors + general_selectors
    
    for selector_type, selector_value in all_selectors:
        try:
            element = wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
            element.click()
            time.sleep(2)
            return True
        except TimeoutException:
            continue
        except Exception:
            continue
    
    return False

def perform_search():
    print("Starting the LuxTimes Search Scraper...")
    create_output_directory()
    
    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--ignore-certificate-errors")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-web-security")
    edge_options.add_argument("--disable-features=VizDisplayCompositor")
    edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver_service = EdgeService(executable_path="msedgedriver.exe")
    driver = webdriver.Edge(service=driver_service, options=edge_options)
    wait = WebDriverWait(driver, 10)
    
    all_results = []

    try:
        for i, query in enumerate(SEARCH_QUERIES, 1):
            print(f"Performing search #{i}: Keywords='{query['keywords']}'")

            try:
                search_url = SEARCH_URL_TEMPLATE.format(query['keywords'])
                driver.get(search_url)
                print(f"Navigated to: {search_url}")
                time.sleep(3)

                handle_cookies(driver, wait)

                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".search-results, .result, .article-item, .search-result, .DefaultTeaser_default-teaser__link__rjMVf")))
                    print("Search results detected")
                    time.sleep(2)
                except TimeoutException:
                    print("Search results did not load properly")

                    continue

                results = driver.find_elements(By.CSS_SELECTOR, "a.DefaultTeaser_default-teaser__link__rjMVf")
                
                if not results:
                    print("No results found for this query.")
                    continue

                print(f"Found {len(results)} results. Processing top {min(len(results), MAX_PAGES_TO_SAVE)}...")
                saved_count = 0

                for j, result in enumerate(results[:MAX_PAGES_TO_SAVE], 1):
                    try:
                        # Extract URL
                        relative_url = result.get_attribute('href')
                        url = urljoin(query['base_url'], relative_url)
                        
                        # Extract title
                        title_element = result.find_element(By.CSS_SELECTOR, ".TeaserContent_teaser-content__title__title__7NXi9")
                        title = title_element.text.strip()
                        
                        # Extract date
                        publication_date = None
                        try:
                            date_element = result.find_element(By.CSS_SELECTOR, "time.DateTime_root__Rlxu5")
                            date_string = date_element.get_attribute('datetime') or date_element.text
                            if date_string:
                                try:
                                    publication_date = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                                except:
                                    publication_date = parse_date(date_string)
                        except:
                            pass

                        if not url or not title:
                            print(f"  Could not extract title or URL from result {j}")
                            continue

                        print(f"  Result {j}:")
                        print(f"  - Title: {title}")
                        print(f"  - URL: {url}")
                        if publication_date:
                            print(f"  - Date: {publication_date.strftime('%Y-%m-%d')}")

                        if save_html_page(url, title, i, j):
                            saved_count += 1
                        
                        # Store result data
                        result_data = {
                            "url": url,
                            "title": title,
                            "date": publication_date.strftime('%Y-%m-%d') if publication_date else None,
                            "keyword": query['keywords']
                        }
                        all_results.append(result_data)
                        
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"  Error processing result {j}")
                        continue

                print(f"Successfully saved {saved_count} pages from this search.")

            except Exception as e:
                print(f"An error occurred during search #{i}: {e}")
                driver.save_screenshot(f"error_screenshot_{i}.png")
                continue
                
    finally:
        driver.quit()
        
        # Save all results to JSON
        json_filepath = os.path.join(OUTPUT_DIR, "scraped_data.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_results)} results to {json_filepath}")

if __name__ == "__main__":
    perform_search()