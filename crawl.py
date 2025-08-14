import os
import json
from urllib.parse import urljoin, urlparse, unquote
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.edge.service import Service as EdgeService

from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from bs4 import BeautifulSoup
import logging as logger    
VA_ENVIRONMENT = "PROD"
# Set up logging
logger.basicConfig(
    level=logger.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

links_requested = set()
redirected_urls = []


def save_content(url, content, folder):
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")
    if not path:
        path = "index.html"
    file_path = os.path.join(folder, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as file:
            file.write(content)
        logger.info(f"Saved: {url}")
    else:
        logger.info("File exists already.")

def download_with_selenium(url, folder):
    """Helper function to download content using Selenium"""
    driver = None
    try:
        # Setup Edge WebDriver
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        service = EdgeService(executable_path="msedgedriver.exe")
        driver = webdriver.Edge(service=service, options=options)        
        
        logger.info(f"Loading page with Selenium: {url}")
        driver.get(url)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        time.sleep(3)
        
        # Get rendered page source
        page_source = driver.page_source.encode('utf-8')
        current_url = driver.current_url
        
        save_content(current_url, page_source, folder)
        return page_source
        
    except Exception as e:
        logger.error(f"Selenium download failed for {url}: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def download_resource(url, folder):
    retries = 5
    try:
        normalized_url = url.encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode URL using latin1 -> utf-8. Using original URL: {url}")
        normalized_url = url
    
    # First, try to determine if we need Selenium (check if it's likely a JavaScript-heavy page)
    try:
        # Quick check - if URL ends with .pdf, use requests directly
        if normalized_url.lower().endswith('.pdf'):
            use_selenium = False
        else:
            # For HTML pages, try Selenium first for better rendering
            use_selenium = True
    except:
        use_selenium = True
    
    if use_selenium:
        # Try Selenium first for HTML content
        selenium_result = download_with_selenium(normalized_url, folder)
        if selenium_result:
            return selenium_result
        # If Selenium fails, fall through to requests
    
    # Fallback to original requests method
    for attempt in range(retries):
        try:
            response = requests.get(normalized_url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                if response.history:
                    logger.info(
                        f"Redirected from {normalized_url} to final URL: {response.url}"
                    )
                    redirect_info = {
                        "original_url": normalized_url,
                    }
                    redirected_urls.append(redirect_info)
                final_url = response.url
                save_content(final_url, response.content, folder)
                return response
            else:
                logger.warning(f"HTTP {response.status_code} for {normalized_url}")
        except Exception as e:
            logger.info(f"Attempt {attempt + 1} failed to download {normalized_url}: {e}")
    
    logger.info(f"Failed to download {normalized_url} after {retries} attempts")
    return None

def crawl_webpage(
    base_url,
    folder,
    dev_mode,
    links_requested=None,
    download_count=None,
    follow_links=True,
):
    if links_requested is None:
        links_requested = set()
    
    if download_count is None:
        download_count = {"count": 0}
    
    limit_dev_document_update_ingest = 10000
    limit = limit_dev_document_update_ingest if dev_mode else None
    
    if not base_url.startswith("https://www.bgl.lu"):
        return
    
    os.makedirs(folder, exist_ok=True)
    
    if base_url in links_requested:
        return
    links_requested.add(base_url)
    
    if limit is not None and download_count["count"] >= limit:
        return
    
    response = download_resource(base_url, folder)
    if not response:
        return
    
    download_count["count"] += 1
    
    if not (base_url.endswith(".html") or base_url.endswith(".pdf")):
        return
    
    if not follow_links:
        return
    
    if base_url.endswith(".html"):
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as e:
            logger.info(f"Failed to parse {base_url}: {e}")
            return
        
        resources = set()
        for tag in soup.find_all(["a", "link", "script"]):
            if tag.name in ["a", "link"]:
                url = tag.get("href")
            else:
                continue
            
            if url:
                full_url = urljoin(base_url, url)
                if full_url.endswith(".html") or full_url.endswith(".pdf"):
                    resources.add(full_url)
        
        for resource in resources:
            if resource not in links_requested:
                crawl_webpage(
                    resource,
                    folder,
                    dev_mode=dev_mode,
                    links_requested=links_requested,
                    download_count=download_count,
                )

def crawl_with_fallback(base_urls, folder, dev_mode, download_count, follow_links):
    for base_url in base_urls:
        try:
            logger.info(f"Attempting to crawl: {base_url}")
            crawl_webpage(
                base_url=base_url,
                folder=folder,
                dev_mode=dev_mode,
                download_count=download_count,
                follow_links=follow_links,
            )
            return
        except Exception as e:
            logger.error(f"Failed to crawl {base_url}: {e}")
    logger.error("All fallback URLs failed. Exiting.")

def scrap_public_site(brute_files_path: str, dev_mode: bool):
    if VA_ENVIRONMENT == "CI":
        base_urls = ["https://www.bgl.lu/fr/particuliers.html","https://www.bgl.lu/de/privatkunden.html","https://www.bgl.lu/en/individuals.html"]
        download_count = {"count": 0}
        for base_url in base_urls:
            crawl_webpage(
                base_url=base_url,
                folder=brute_files_path,
                dev_mode=dev_mode,
                download_count=download_count,
                follow_links=False,
            )
        return
    
    download_count = {"count": 0}
    base_urls = ["https://www.bgl.lu/fr/particuliers.html","https://www.bgl.lu/de/privatkunden.html","https://www.bgl.lu/en/individuals.html"]
    crawl_with_fallback(
        base_urls=base_urls,
        folder=brute_files_path,
        dev_mode=dev_mode,
        download_count=download_count,
        follow_links=True,
    )
    
    # Save redirected URLs to JSON file after crawling is complete
    if redirected_urls:
        filepath = os.path.join(brute_files_path, "redirected_urls.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(redirected_urls, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(redirected_urls)} redirected URLs to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save redirected URLs to JSON: {e}")
    else:
        logger.info("No redirected URLs to save")

if __name__ == "__main__":
    brute_files_path = "brute_files"
    # scrap_public_site(brute_files_path, dev_mode=False)
    download_resource("http://bgl.lu/en/individuals/contact/branch-locator.html", brute_files_path)