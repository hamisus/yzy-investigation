"""
YEWS.news image scraper module for the YzY investigation project.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from playwright.async_api import Page, TimeoutError, async_playwright
from urllib.parse import urljoin
import asyncio
import aiohttp

from yzy_investigation.core.base_pipeline import BasePipeline
from yzy_investigation.core.log_manager import LogManager
from yzy_investigation.core.data_manager import DataManager

# Create a named logger
logger = logging.getLogger("yzy_investigation.web_scraper")

@dataclass
class Article:
    """Data class to store article information."""
    title: str
    text: str
    images: List[Dict[str, str]]  # List of dicts with 'url' and 'alt' keys
    sources: List[Dict[str, str]]  # List of dicts with 'url' and 'text' keys
    time_section: str
    date: str


class YewsScraper(BasePipeline):
    """A scraper class to download images from YEWS.news."""
    
    def __init__(self, input_path: Optional[Path] = None, output_path: Optional[Path] = None) -> None:
        """
        Initialize the scraper.
        
        Args:
            input_path: Not used in this implementation
            output_path: Optional path for output data. If None, uses data/raw/yews/
        """
        # We'll set the output path later when we know the article date
        super().__init__(input_path or Path("data/raw"), output_path or Path("data/raw/yews"))
        
        self.base_url = "https://www.yews.news"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'YzY Investigation Bot/1.0'
        }
        self.visited_urls: Set[str] = set()
        self.articles: List[Article] = []
        self.article_counter: Dict[str, int] = {}
        self.log_manager = LogManager()
        self.data_manager = DataManager()
        self.current_date: Optional[str] = None
        
        # Initialize logger
        self.logger = logger

    def _extract_date_from_url(self, url: str) -> Optional[str]:
        """
        Extract date from YEWS image URL.
        
        Args:
            url: Image URL from YEWS
            
        Returns:
            Date string in YYYY-MM-DD format or None if not found
        """
        # Match pattern like "3-27-25" in URLs
        match = re.search(r'/(\d{1,2}-\d{1,2}-\d{2})/', url)
        if match:
            # Convert from M-D-YY to YYYY-MM-DD
            date_str = match.group(1)
            month, day, year = map(int, date_str.split('-'))
            year = 2000 + year  # Assume 20xx for two-digit year
            return f"{year}-{month:02d}-{day:02d}"
        return None

    def _get_safe_filename(self, text: str, max_length: int = 50) -> str:
        """
        Create a safe filename from text.
        
        Args:
            text: Text to convert to safe filename
            max_length: Maximum length of the resulting filename
            
        Returns:
            A safe filename string
        """
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', text)
        safe_name = re.sub(r'_+', '_', safe_name)
        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length].rsplit('_', 1)[0]
        return safe_name.strip('_')

    def _get_article_dir(self, article: Article) -> Path:
        """
        Get the directory path for an article's files.
        
        Args:
            article: Article object
            
        Returns:
            Path to the article directory
        """
        if article.time_section not in self.article_counter:
            self.article_counter[article.time_section] = 0
        self.article_counter[article.time_section] += 1
        
        safe_title = self._get_safe_filename(article.title)
        # Use the article's date for the directory structure
        date_dir = self.output_path / article.date
        article_dir = date_dir / f"{self.article_counter[article.time_section]:02d}_{article.time_section}_{safe_title}"
        article_dir.mkdir(parents=True, exist_ok=True)
        return article_dir

    def save_article_metadata(self, article: Article) -> Path:
        """
        Save article metadata to a JSON file.
        
        Args:
            article: Article object containing metadata
            
        Returns:
            Path to the article directory
        """
        article_dir = self._get_article_dir(article)
        metadata_file = article_dir / "metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(article), f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved metadata for article: {article.title}")
        return article_dir

    def download_image(self, url: str, article_dir: Path, img_index: int) -> None:
        """
        Download an image and save with contextual filename.
        
        Args:
            url: URL of the image to download
            article_dir: Directory to save the image in
            img_index: Index of the image within the article
        """
        if url in self.visited_urls:
            self.logger.debug(f"Skipping already downloaded image: {url}")
            return
            
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            extension = url.split('.')[-1].split('?')[0] if '.' in url else 'jpg'
            filename = f"image_{img_index:02d}.{extension}"
            filepath = article_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            self.logger.info(f"Successfully downloaded: {filename}")
            self.visited_urls.add(url)
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download image from {url}: {e}")
        except IOError as e:
            self.logger.error(f"Failed to save image from {url}: {e}")

    async def click_time_button(self, page: Page, time_text: str) -> bool:
        """
        Click a time button (10AM, 3PM, 8PM) on the main page.
        
        Args:
            page: Playwright page object
            time_text: The text of the time button to click
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            button = await page.wait_for_selector(f"text={time_text}", timeout=3000)
            if button:
                await button.click()
                await page.wait_for_timeout(1000)
                return True
            return False
        except TimeoutError:
            self.logger.warning(f"Could not find {time_text} button")
            return False
        except Exception as e:
            self.logger.error(f"Error clicking {time_text} button: {e}")
            return False

    async def expand_all_articles(self, page: Page) -> None:
        """
        Expand all article sections on the page.
        
        Args:
            page: Playwright page object
        """
        try:
            await page.wait_for_timeout(1000)
            
            expand_buttons = await page.query_selector_all('div[role="button"]')
            self.logger.debug(f"Found {len(expand_buttons)} expand buttons")
            
            click_tasks = []
            for button in expand_buttons:
                if await button.is_visible():
                    click_tasks.append(button.click())
            
            if click_tasks:
                self.logger.debug(f"Clicking {len(click_tasks)} buttons in parallel")
                await asyncio.gather(*click_tasks)
                await page.wait_for_timeout(1000)
            
        except Exception as e:
            self.logger.warning(f"Failed to expand articles: {e}")

    def extract_article_data(self, html_content: str, time_section: str) -> List[Article]:
        """
        Extract article data from the page.
        
        Args:
            html_content: HTML content of the page
            time_section: Current time section being processed
            
        Returns:
            List of Article objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []
        
        article_sections = soup.find_all('div', class_=lambda x: x and 'flex flex-col' in x and 'opacity-0' in x)
        
        for section in article_sections:
            title_elem = section.find('span', class_=lambda x: x and 'cursor-pointer' in x and 'break-words' in x)
            title = title_elem.get_text(strip=True) if title_elem else "Untitled"
            
            text_elem = section.find('div', class_=lambda x: x and 'break-words' in x and 'whitespace-pre-line' in x)
            text = text_elem.get_text(strip=True) if text_elem else ""
            
            images = []
            article_date = None
            img_elements = section.find_all('img')
            for img in img_elements:
                src = img.get('src', '')
                if not src or '.svg' in src.lower():
                    continue
                    
                src = urljoin(self.base_url, src)
                
                # Extract date from first valid image URL
                if not article_date:
                    article_date = self._extract_date_from_url(src)
                
                img_title = ''
                title_span = img.find_next('span', class_=lambda x: x and 'text-gray-500' in x and 'break-words' in x)
                if title_span:
                    img_title = title_span.get_text(strip=True)
                
                images.append({
                    'url': src,
                    'alt': img.get('alt', ''),
                    'title': img_title
                })
            
            sources = []
            source_links = section.find_all('a', href=True)
            for link in source_links:
                href = link.get('href')
                if href:
                    href = urljoin(self.base_url, href)
                    sources.append({
                        'url': href,
                        'text': link.get_text(strip=True)
                    })
            
            if images and article_date:  # Only add articles with images and valid date
                article = Article(
                    title=title,
                    text=text,
                    images=images,
                    sources=sources,
                    time_section=time_section,
                    date=article_date
                )
                articles.append(article)
        
        return articles

    async def process_time_section(self, page: Page, time_text: str) -> None:
        """
        Process a time section of the page.
        
        Args:
            page: Playwright page object
            time_text: The time section to process
        """
        self.logger.info(f"Processing time section: {time_text}")
        
        try:
            self.article_counter[time_text] = 0
            
            if not await self.click_time_button(page, time_text):
                self.logger.error(f"Failed to click {time_text} button")
                return
                
            await self.expand_all_articles(page)
            
            content = await page.content()
            articles = self.extract_article_data(content, time_text)
            self.logger.info(f"Found {len(articles)} articles with images in {time_text} section")
            
            tasks = []
            for article in articles:
                article_dir = self.save_article_metadata(article)
                for i, img in enumerate(article.images, 1):
                    tasks.append(asyncio.create_task(self._download_image_async(img['url'], article_dir, i)))
            
            if tasks:
                self.logger.info(f"Downloading {len(tasks)} images...")
                await asyncio.gather(*tasks)
                
        except Exception as e:
            self.logger.error(f"Failed to process time section {time_text}: {e}")

    async def _download_image_async(self, url: str, article_dir: Path, img_index: int) -> None:
        """
        Download an image asynchronously.
        
        Args:
            url: URL of the image to download
            article_dir: Directory to save the image in
            img_index: Index of the image within the article
        """
        # Use the full URL as the unique identifier
        if url in self.visited_urls:
            self.logger.debug(f"Skipping already downloaded image: {url}")
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    
                    extension = url.split('.')[-1].split('?')[0] if '.' in url else 'jpg'
                    filename = f"image_{img_index:02d}.{extension}"
                    filepath = article_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    self.logger.info(f"Successfully downloaded: {filename}")
                    self.visited_urls.add(url)
            
        except Exception as e:
            self.logger.error(f"Failed to download image from {url}: {e}")

    async def _run_async(self) -> Dict[str, Any]:
        """Execute the complete scraping process asynchronously."""
        self.logger.info("Starting YEWS.news image scraping process...")
        
        start_time = datetime.now()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 390, 'height': 844},
                device_scale_factor=2
            )
            
            page = await context.new_page()
            
            try:
                self.logger.info(f"Navigating to {self.base_url}")
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                for time_text in ["10AM", "3PM", "8PM"]:
                    await self.process_time_section(page, time_text)
                    await page.goto(self.base_url)
                    await page.wait_for_timeout(1000)
                
            finally:
                await browser.close()
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare results
        results = {
            "status": "success",
            "articles_processed": len(self.articles),
            "images_downloaded": len(self.visited_urls),
            "execution_time_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        # Log structured event
        self.log_manager.log_event("scrape_completed", {
            "source": "yews.news",
            "articles_processed": len(self.articles),
            "images_downloaded": len(self.visited_urls),
            "execution_time_seconds": duration,
            "article_dates": list(set(article.date for article in self.articles))
        })
        
        self.logger.info(f"Scraping completed: {len(self.articles)} articles with {len(self.visited_urls)} images in {duration:.2f} seconds")
        return results

    def validate_input(self) -> bool:
        """
        Validate input parameters.
        
        Returns:
            bool: Always True as this scraper doesn't require input validation
        """
        return True

    def run(self) -> Dict[str, Any]:
        """
        Run the scraper.
        
        Returns:
            Dict containing results and metadata
        """
        return asyncio.run(self._run_async()) 