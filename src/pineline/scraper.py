from bs4 import BeautifulSoup
from abc import abstractmethod
import httpx
from typing import Any
from playwright.sync_api import sync_playwright, Page

class BaseScraper:
    @abstractmethod
    def fetch(self, url: str) -> Any:
        pass
    
    @abstractmethod
    def parse(self, html:str) -> Any:
        pass
    
    @abstractmethod
    def extract(self, parsed_data) -> Any:
        pass
    
    def scrape(self, url:str) -> Any:
        raw_data = self.fetch(url)
        parsed_data = self.parse(raw_data)
        return self.extract(parsed_data)
    
class StaticScraper(BaseScraper):
    def __init__(self):
        # Stored common config
        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }
        self.timeout = 10

    def fetch(self, url: str) -> str:
        with httpx.Client(headers=self.headers, timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    def parse(self, html:str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def extract(self, parsed_data: BeautifulSoup) -> str:
        # Extract text content from the parsed HTML
        return parsed_data.get_text()
    
class DynamicScraper(BaseScraper):
    def __init__(self, headless: bool=True, timeout: int=30000):
        self.headless = headless
        self.timeout = timeout
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def fetch(self, url: str) -> Page:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

        self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
        # Wait for content to load
        self.page.wait_for_timeout(3000)  # 3 second wait
        return self.page

    def parse(self, page: Page):
        # Page is already loaded from fetch method
        return page

    def extract(self, page: Page):
        # Extract text content from the page
        return page.content()

    def scrape(self, url: str):
        try:
            raw_data = self.fetch(url)
            parsed_data = self.parse(raw_data)
            return self.extract(parsed_data)
        finally:
            self.close()

    def close(self):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
            
class ScraperFactory:
    @staticmethod
    def detect_page_type(url: str) -> str:
        try:
            response = httpx.get(url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            html = response.text.lower()
            
            dynamic_indicators = [
                "id='__next'",
                "id='__nuxt'",
                "id='root'",
                "id='app'",
                "window.__initial_state__",
                "window.__nuxt__",
                "webpack",
                "react",
                "vue",
                "angular"
            ]
            
            score = 0
            
            for indicator in dynamic_indicators:
                if indicator in html:
                    score += 1
            if html.count("<script") > 15:
                score += 2
            if len(html.strip()) < 5000:
                score += 1
            
            return "dynamic" if score > 3 else "static"
        
        except Exception:
            return "dynamic"
        
    @staticmethod
    def get_scraper(url: str):
        page_type = ScraperFactory.detect_page_type(url)
        
        if page_type == "dynamic":
            return DynamicScraper()
        return StaticScraper()
    
if __name__ == "__main__":
    url = "https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/"
    scraper = ScraperFactory.get_scraper(url)
    result = scraper.scrape(url)

    print(type(scraper).__name__)
    if result:
        print(f"Scraping successful! Content length: {len(result)} characters")
        # Print first 500 characters to avoid console encoding issues
        print(f"Preview: {result[:500]}...")
    else:
        print("Scraping failed - no content returned")