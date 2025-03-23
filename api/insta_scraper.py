import requests
from bs4 import BeautifulSoup
import json
import re
import argparse
import os
import time
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
import undetected_chromedriver as uc


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstagramScraper:
    def __init__(self, headless=True):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Set up undetected-chromedriver instead of regular Selenium
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={self.headers['User-Agent']}")
        
        # Create undetected-chromedriver instance
        self.driver = uc.Chrome(options=options, version_main=134)  # Specify a Chrome version that matches your installed Chrome
        
        # Set up wait
        self.wait = WebDriverWait(self.driver, 10)
        
    def extract_username_from_url(self, url):
        """Extract username from Instagram URL"""
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        return path_parts[0] if path_parts else None
    
    def get_profile_info(self, profile_url):
        """Get profile information using Selenium"""
        username = self.extract_username_from_url(profile_url)
        if not username:
            logger.error("Could not extract username from URL")
            return None
            
        logger.info(f"Retrieving profile for: {username}")
        
        try:
            # First try loading the Instagram page directly
            self.driver.get("https://www.instagram.com/")
            time.sleep(3)  # Wait longer for initial page load
            
            # Dump page source to see what we're dealing with
            with open("debug_initial_page.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.info(f"Saved initial page source to debug_initial_page.html")
            
            # Handle the Google popup - more comprehensive approach
            try:
                # Look for various buttons that might close the popup
                popup_close_selectors = [
                    "//div[@role='dialog']//button[contains(@aria-label, 'Close')]",
                    "//button[contains(@aria-label, 'Close')]",
                    "//svg[contains(@aria-label, 'Close')]/..",
                    "//div[@role='dialog']//div[@role='button']",
                    "//div[@role='presentation']//button"
                ]
                
                for selector in popup_close_selectors:
                    try:
                        close_buttons = self.driver.find_elements(By.XPATH, selector)
                        for button in close_buttons:
                            try:
                                button.click()
                                logger.info(f"Clicked potential popup close button using selector: {selector}")
                                time.sleep(1)
                                break
                            except:
                                continue
                    except:
                        continue
            except Exception as e:
                logger.debug(f"Error handling popup: {e}")
            
            # Check for login page
            current_url = self.driver.current_url
            logger.info(f"Current URL after initial handling: {current_url}")
            
            # Better login page detection
            is_login_page = ("login" in current_url or 
                            "accounts/login" in current_url or 
                            "sign in" in self.driver.page_source.lower() or
                            "log in" in self.driver.page_source.lower())
            
            # Handle login if needed
            if is_login_page:
                logger.info("Login page detected, attempting to login")
                try:
                    # Wait for username field with longer timeout
                    username_input = WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@name='username']"))
                    )
                    
                    # Wait for password field
                    password_input = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
                    )
                    
                    # Clear fields first
                    username_input.clear()
                    password_input.clear()
                    
                    # Replace with your Instagram credentials
                    username_input.send_keys("tostiffent")
                    time.sleep(1)  # Small pause between inputs
                    password_input.send_keys("tosti@1234")
                    
                    # Find and click login button - more reliable selector
                    login_button_selectors = [
                        "//button[@type='submit']",
                        "//button[contains(text(), 'Log in')]",
                        "//button[contains(text(), 'Sign in')]",
                        "//button[contains(@class, 'login')]"
                    ]
                    
                    for selector in login_button_selectors:
                        try:
                            login_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                            login_button.click()
                            logger.info(f"Clicked login button using selector: {selector}")
                            break
                        except:
                            continue
                    
                    # Wait longer for login to complete
                    time.sleep(8)
                    
                    # Verify login success
                    if "login" not in self.driver.current_url and "accounts/login" not in self.driver.current_url:
                        logger.info("Login successful!")
                    else:
                        logger.error("Login may have failed - still on login page")
                        
                    # Save page after login attempt for debugging
                    with open("debug_after_login.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    logger.info("Saved post-login page source for debugging")
                    
                    # Handle additional dialogs that might appear after login
                    dialogs = [
                        {"text": "Not Now", "timeout": 5},
                        {"text": "Not now", "timeout": 5},
                        {"text": "Save Info", "timeout": 5},
                        {"text": "Cancel", "timeout": 5}
                    ]
                    
                    for dialog in dialogs:
                        try:
                            button = WebDriverWait(self.driver, dialog["timeout"]).until(
                                EC.element_to_be_clickable((By.XPATH, f"//button[contains(text(), '{dialog['text']}')]"))
                            )
                            button.click()
                            logger.info(f"Handled '{dialog['text']}' dialog")
                            time.sleep(1)
                        except:
                            logger.debug(f"No '{dialog['text']}' dialog appeared or couldn't click it")
                except Exception as e:
                    logger.error(f"Login process failed: {e}")
            
            # Now navigate to the profile page
            logger.info(f"Navigating to profile URL: {profile_url}")
            self.driver.get(profile_url)
            time.sleep(5)  # Give more time for page to load
            
            # Check if we got redirected to login again
            if "login" in self.driver.current_url or "accounts/login" in self.driver.current_url:
                logger.error("Got redirected to login page again - login likely failed")
            
            # Extract profile data
            profile_data = {
                "username": username,
                "profile_url": profile_url,
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Save page source for debugging
            with open(f"debug_{username}_page.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.info(f"Saved page source to debug_{username}_page.html for inspection")
            
            # Get profile picture - use multiple selectors to try to find it
            profile_selectors = [
                "//img[@alt and contains(@alt, 'profile picture')]", 
                "//header//img", 
                "//div[contains(@class, 'Profile')]//img",
                "//div[contains(@class, 'profile')]//img",
                "//div[@role='presentation']//img"
            ]
            
            for selector in profile_selectors:
                try:
                    profile_pic_element = self.driver.find_element(By.XPATH, selector)
                    profile_data["profile_picture_url"] = profile_pic_element.get_attribute("src")
                    logger.info(f"Profile picture URL found using selector: {selector}")
                    break
                except NoSuchElementException:
                    continue
            
            if "profile_picture_url" not in profile_data:
                logger.warning("Could not extract profile picture with any selector")
                profile_data["profile_picture_url"] = None
            
            # Try different approaches to get user information
            # Method 1: Extract from meta tags (most reliable for public profiles)
            try:
                # Extract from meta description tag
                meta_description = self.driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content")
                if meta_description:
                    logger.info(f"Found meta description: {meta_description[:100]}...")
                    
                    # Extract full name and username from meta description
                    # Pattern: "... from Full Name (@username)"
                    name_username_match = re.search(r'from\s+([^(]+)\s+\(@([^)]+)\)', meta_description)
                    if name_username_match:
                        profile_data["full_name"] = name_username_match.group(1).strip()
                        # Username is already extracted from URL, but we can verify it
                        extracted_username = name_username_match.group(2).strip()
                        if extracted_username != username:
                            logger.warning(f"Username mismatch: {username} vs {extracted_username}")
                    
                    # Extract metrics from meta description
                    # Example: "1.2M Followers, 100 Following, 500 Posts"
                    followers_match = re.search(r'([\d,.]+[kmbt]?)\s+(?:followers|follower)', meta_description.lower())
                    if followers_match:
                        profile_data["followers_count"] = followers_match.group(1)
                        logger.info(f"Extracted followers count from meta: {profile_data['followers_count']}")
                        
                    following_match = re.search(r'([\d,.]+[kmbt]?)\s+following', meta_description.lower())
                    if following_match:
                        profile_data["following_count"] = following_match.group(1)
                        logger.info(f"Extracted following count from meta: {profile_data['following_count']}")
                    
                    posts_match = re.search(r'([\d,.]+[kmbt]?)\s+(?:posts|post)', meta_description.lower())
                    if posts_match:
                        profile_data["posts_count"] = posts_match.group(1)
                        logger.info(f"Extracted posts count from meta: {profile_data['posts_count']}")
                
                # Extract from og:title meta tag (contains profile name)
                try:
                    og_title = self.driver.find_element(By.XPATH, "//meta[@property='og:title']").get_attribute("content")
                    if og_title and not profile_data.get("full_name"):
                        # Pattern: "Username on Instagram: ..." or "Full Name (@username) on Instagram"
                        name_match = re.search(r'^([^(]+)(?:\s+\(@[^)]+\))?\s+on\s+Instagram', og_title)
                        if name_match:
                            profile_data["full_name"] = name_match.group(1).strip()
                            logger.info(f"Extracted full name from og:title: {profile_data['full_name']}")
                except:
                    logger.debug("No og:title meta tag found or couldn't extract name")
                    
                # Extract from og:description meta tag (may contain bio)
                try:
                    og_description = self.driver.find_element(By.XPATH, "//meta[@property='og:description']").get_attribute("content")
                    if og_description:
                        # The og:description often contains likes, comments, and the post caption
                        # We can extract the bio if it's a profile page
                        if "likes" not in og_description.lower() and "comments" not in og_description.lower():
                            # Don't use meta description as bio if it contains follower/following counts
                            # This is likely the Instagram metadata, not the actual bio
                            if not (re.search(r'followers', og_description.lower()) and 
                                   re.search(r'following', og_description.lower()) and 
                                   re.search(r'posts', og_description.lower())):
                                profile_data["bio"] = og_description.strip()
                                logger.info(f"Extracted bio from og:description: {profile_data['bio'][:50]}...")
                except:
                    logger.debug("No og:description meta tag found or couldn't extract bio")
                    
                # Try to extract bio from the profile page directly - more reliable method
                try:
                    # Look for bio section which often appears after the username/full name
                    bio_selectors = [
                        "//section//h1/following-sibling::div[not(contains(@class, 'follow'))]//span",
                        "//header//section//div[not(contains(@class, 'follow'))]/span",
                        "//header//section//div[not(contains(@class, 'follow'))]/div",
                        "//div[contains(@class, 'profile')]/div[not(contains(text(), 'followers') or contains(text(), 'following'))]",
                        "//div[contains(@class, 'biography')]",
                        "//div[contains(@class, 'QGPASd')]",  # Common Instagram bio class
                        "//div[contains(@class, '_aa_c')]",   # Another Instagram bio class
                        "//div[contains(@class, 'profile')]//div[not(contains(@class, 'follow'))][string-length(text()) > 10]"
                    ]
                    
                    for selector in bio_selectors:
                        bio_elements = self.driver.find_elements(By.XPATH, selector)
                        for bio_elem in bio_elements:
                            bio_text = bio_elem.text.strip()
                            # Skip if text is too short, contains metrics, or is likely a button
                            if (bio_text and 
                                len(bio_text) > 10 and
                                bio_text != profile_data.get("full_name", "") and
                                "follow" not in bio_text.lower() and
                                not (re.search(r'followers', bio_text.lower()) and 
                                     re.search(r'following', bio_text.lower()))):
                                profile_data["bio"] = bio_text
                                logger.info(f"Extracted bio directly from profile: {profile_data['bio'][:50]}...")
                                break
                        if profile_data.get("bio") and profile_data["bio"] != "Follow":
                            break
                except Exception as e:
                    logger.debug(f"Error extracting bio directly: {e}")
            except Exception as e:
                logger.warning(f"Error extracting from meta tags: {e}")
            
            # Method 2: Look for specific sections in the DOM
            try:
                section_elements = self.driver.find_elements(By.XPATH, "//section | //header | //div[@role='tablist']/parent::*")
                for section in section_elements:
                    try:
                        section_text = section.text
                        if username.lower() in section_text.lower() or "followers" in section_text.lower():
                            # Parse full name if not already found
                            if not profile_data.get("full_name"):
                                name_candidates = section.find_elements(By.XPATH, ".//h1 | .//h2 | .//span[contains(@class, 'full')]")
                                if name_candidates:
                                    profile_data["full_name"] = name_candidates[0].text.strip()
                                    logger.info(f"Extracted full name from section: {profile_data['full_name']}")
                                
                            # Parse bio if not already found
                            if not profile_data.get("bio"):
                                # Try multiple selectors for bio
                                bio_selectors = [
                                    ".//div[text() and not(child::*) and string-length(text()) > 5 and not(contains(text(), 'followers') and contains(text(), 'following') and contains(text(), 'posts'))]",
                                    ".//span[contains(@class, 'bio')]",
                                    ".//div[contains(@class, 'bio')]",
                                    ".//h1/following-sibling::div[not(contains(@class, 'follow'))]",
                                    ".//div[contains(@class, 'profile')]/div[not(contains(text(), 'followers') or contains(text(), 'following'))]",
                                    ".//*[contains(text(), '.com/') or contains(text(), 'http')]/parent::div"
                                ]
                                
                                for bio_selector in bio_selectors:
                                    bio_candidates = section.find_elements(By.XPATH, bio_selector)
                                    if bio_candidates:
                                        for bio in bio_candidates:
                                            bio_text = bio.text.strip()
                                            # Skip if text contains follower/following metrics or is too short
                                            if (bio_text and 
                                                bio_text != profile_data.get("full_name", "") and 
                                                len(bio_text) > 5 and
                                                not (re.search(r'[\d,.]+\s+followers', bio_text.lower()) and 
                                                     re.search(r'[\d,.]+\s+following', bio_text.lower()) and
                                                     re.search(r'[\d,.]+\s+posts', bio_text.lower()))):
                                                profile_data["bio"] = bio_text
                                                logger.info(f"Extracted bio from section: {profile_data['bio'][:50]}...")
                                                break
                                        if profile_data.get("bio"):
                                            break
                                        
                            # Parse metrics (posts, followers, following) if not already found
                            if not all(k in profile_data for k in ["posts_count", "followers_count", "following_count"]):
                                metrics_selectors = [
                                    ".//span[contains(text(), 'post') or contains(text(), 'follower') or contains(text(), 'following')]",
                                    ".//li[contains(text(), 'post') or contains(text(), 'follower') or contains(text(), 'following')]",
                                    ".//a[contains(text(), 'post') or contains(text(), 'follower') or contains(text(), 'following')]",
                                    ".//div[contains(text(), 'post') or contains(text(), 'follower') or contains(text(), 'following')]"
                                ]
                                
                                for selector in metrics_selectors:
                                    metrics = section.find_elements(By.XPATH, selector)
                                    for metric in metrics:
                                        metric_text = metric.text.lower()
                                        if 'post' in metric_text and not profile_data.get("posts_count"):
                                            profile_data["posts_count"] = self.extract_number(metric_text)
                                            logger.info(f"Extracted posts count from section: {profile_data['posts_count']}")
                                        elif 'follower' in metric_text and not profile_data.get("followers_count"):
                                            profile_data["followers_count"] = self.extract_number(metric_text)
                                            logger.info(f"Extracted followers count from section: {profile_data['followers_count']}")
                                        elif 'following' in metric_text and not profile_data.get("following_count"):
                                            profile_data["following_count"] = self.extract_number(metric_text)
                                            logger.info(f"Extracted following count from section: {profile_data['following_count']}")
                                    
                                    if all(k in profile_data for k in ["posts_count", "followers_count", "following_count"]):
                                        break
                            
                            # If we found most of the data, we can break
                            if all(k in profile_data for k in ["full_name", "bio", "followers_count"]):
                                break
                    except Exception as e:
                        logger.debug(f"Error processing section: {e}")
                        continue
                        
                logger.info("Processed profile sections for information extraction")
            except Exception as e:
                logger.warning(f"Could not extract profile info from sections: {e}")
            
            # Method 3: Look for specific metrics directly if still missing
            if not all(k in profile_data for k in ["posts_count", "followers_count", "following_count"]):
                try:
                    metrics_elements = self.driver.find_elements(By.XPATH, "//li | //div[contains(text(), 'follower') or contains(text(), 'following') or contains(text(), 'post')]")
                    for element in metrics_elements:
                        text = element.text.lower()
                        if 'post' in text and not profile_data.get("posts_count"):
                            profile_data["posts_count"] = self.extract_number(text)
                            logger.info(f"Extracted posts count from direct selector: {profile_data['posts_count']}")
                except Exception as e:
                        logger.warning(f"Error extracting metrics directly: {e}")
            # Get website
            try:
                website_element = self.driver.find_element(By.XPATH, "//a[contains(@href, 'http') and not(contains(@href, 'instagram.com'))]")
                profile_data["website"] = website_element.get_attribute("href")
                logger.info("Website found")
            except:
                profile_data["website"] = None
                logger.debug("No website found")
            
            # Get recent posts - first identify the post grid
            try:
                # Wait for posts to load if they're not immediately visible
                time.sleep(3)
                # Get all posts by default (post_count=None)
                profile_data["recent_posts"] = self.get_recent_posts()
                logger.info(f"Retrieved {len(profile_data['recent_posts'])} posts from profile")
            except Exception as e:
                logger.error(f"Error getting recent posts: {e}")
                profile_data["recent_posts"] = []
                
            # Get followers information (limited to 5)
            try:
                followers = self.get_followers(username, max_count=5)
                profile_data["followers"] = followers
                logger.info(f"Retrieved {len(followers)} followers")
            except Exception as e:
                logger.error(f"Error getting followers: {e}")
                profile_data["followers"] = []
                
            # Get following information (limited to 5)
            try:
                following = self.get_following(username, max_count=5)
                profile_data["following"] = following
                logger.info(f"Retrieved {len(following)} following")
            except Exception as e:
                logger.error(f"Error getting following: {e}")
                profile_data["following"] = []
            
            # If we couldn't get any data, mark this as probably login-required
            if not profile_data.get("followers_count") and not profile_data.get("posts_count"):
                profile_data["requires_login"] = True
                logger.warning("This profile may require login to view")
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error scraping profile: {e}")
            return None
        
    def extract_number(self, text):
        """Extract number from text like '1,234 followers' or '1.2M followers'"""
        # Handle abbreviated numbers like 1.2M, 4.5K, etc.
        if not text:
            return "0"
            
        abbr_match = re.search(r'([\d,.]+)([kmbt])', text.lower())
        if abbr_match:
            num, unit = abbr_match.groups()
            num = float(num.replace(',', ''))
            if unit == 'k':
                return str(int(num * 1000))
            elif unit == 'm':
                return str(int(num * 1000000))
            elif unit == 'b':
                return str(int(num * 1000000000))
            elif unit == 't':
                return str(int(num * 1000000000000))
        
        # Handle regular numbers
        number = re.search(r'([\d,]+)', text)
        if number:
            return number.group(1).replace(',', '')
        return "0"
        
    def get_followers(self, username, max_count=5):
        """Get followers information
        
        Args:
            username: Instagram username
            max_count: Maximum number of followers to retrieve
        """
        logger.info(f"Attempting to extract followers for {username} (max: {max_count})")
        followers = []
        
        try:
            # Navigate to followers page
            followers_url = f"https://www.instagram.com/{username}/followers/"
            logger.info(f"Navigating to followers URL: {followers_url}")
            self.driver.get(followers_url)
            time.sleep(5)  # Wait for page to load
            
            # Check if we're on the right page
            if "followers" not in self.driver.current_url.lower() and "followers" not in self.driver.page_source.lower():
                logger.warning("Could not access followers page, trying alternative approach")
                
                # Try clicking on followers count link from profile page
                self.driver.get(f"https://www.instagram.com/{username}/")
                time.sleep(3)
                
                # Try different selectors for followers link
                followers_link_selectors = [
                    "//a[contains(@href, '/followers')]|//a[contains(text(), 'followers')]|//span[contains(text(), 'followers')]/parent::a",
                    "//li/a[contains(text(), 'follower')]|//div[@role='button' and contains(text(), 'follower')]",
                    "//div[contains(text(), 'followers')]"
                ]
                
                clicked = False
                for selector in followers_link_selectors:
                    try:
                        followers_elements = self.driver.find_elements(By.XPATH, selector)
                        if followers_elements:
                            for elem in followers_elements:
                                try:
                                    elem.click()
                                    logger.info(f"Clicked followers link using selector: {selector}")
                                    clicked = True
                                    time.sleep(5)  # Wait for modal to open
                                    break
                                except Exception as e:
                                    logger.debug(f"Could not click element: {e}")
                                    continue
                        if clicked:
                            break
                    except Exception as e:
                        logger.debug(f"Error with selector {selector}: {e}")
                        continue
            
            # Look for the followers list - try multiple selectors
            followers_list_selectors = [
                "//div[@role='dialog']//div[@role='tablist']/following-sibling::div//div[@role='button']",
                "//div[@role='dialog']//ul/li",
                "//div[contains(@class, 'followers')]//ul/li",
                "//div[@role='presentation']//ul/li",
                "//div[contains(@aria-label, 'Followers')]//ul/li"
            ]
            
            follower_elements = []
            for selector in followers_list_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements and len(elements) > 0:
                        logger.info(f"Found {len(elements)} follower elements using selector: {selector}")
                        follower_elements = elements[:max_count]  # Limit to max_count
                        break
                except Exception as e:
                    logger.debug(f"Error with follower list selector {selector}: {e}")
                    continue
            
            if not follower_elements:
                logger.warning("Could not find follower elements with any selector")
                return followers
            
            # Extract information from each follower element
            for i, elem in enumerate(follower_elements):
                try:
                    follower_data = {}
                    
                    # Extract username
                    username_selectors = [
                        ".//a[contains(@href, '/')]|.//span[contains(@class, 'username')]",
                        ".//div[contains(@class, 'username')]",
                        ".//span[contains(@class, 'notranslate')]"
                    ]
                    
                    for selector in username_selectors:
                        try:
                            username_elem = elem.find_element(By.XPATH, selector)
                            username_text = username_elem.text
                            if username_text and len(username_text) > 0:
                                follower_data["username"] = username_text
                                # Get profile URL
                                try:
                                    href = username_elem.get_attribute("href")
                                    if href and "instagram.com" in href:
                                        follower_data["profile_url"] = href
                                    else:
                                        follower_data["profile_url"] = f"https://www.instagram.com/{username_text}/"
                                except:
                                    follower_data["profile_url"] = f"https://www.instagram.com/{username_text}/"
                                break
                        except:
                            continue
                    
                    # If we couldn't find username with selectors, try to extract from element text
                    if "username" not in follower_data:
                        elem_text = elem.text
                        if elem_text:
                            lines = elem_text.split('\n')
                            if lines and len(lines) > 0:
                                follower_data["username"] = lines[0].strip()
                                follower_data["profile_url"] = f"https://www.instagram.com/{follower_data['username']}/"
                    
                    # Extract profile picture
                    try:
                        img_elem = elem.find_element(By.XPATH, ".//img")
                        img_src = img_elem.get_attribute("src")
                        if img_src:
                            follower_data["profile_picture_url"] = img_src
                    except:
                        follower_data["profile_picture_url"] = None
                    
                    # Only add if we have at least a username
                    if "username" in follower_data:
                        followers.append(follower_data)
                        logger.info(f"Extracted follower {i+1}: {follower_data['username']}")
                except Exception as e:
                    logger.debug(f"Error extracting follower {i}: {e}")
            
            return followers
            
        except Exception as e:
            logger.error(f"Error getting followers: {e}")
            return followers
    
    def get_following(self, username, max_count=5):
        """Get following information
        
        Args:
            username: Instagram username
            max_count: Maximum number of following to retrieve
        """
        logger.info(f"Attempting to extract following for {username} (max: {max_count})")
        following = []
        
        try:
            # Navigate to following page
            following_url = f"https://www.instagram.com/{username}/following/"
            logger.info(f"Navigating to following URL: {following_url}")
            self.driver.get(following_url)
            time.sleep(5)  # Wait for page to load
            
            # Check if we're on the right page
            if "following" not in self.driver.current_url.lower() and "following" not in self.driver.page_source.lower():
                logger.warning("Could not access following page, trying alternative approach")
                
                # Try clicking on following count link from profile page
                self.driver.get(f"https://www.instagram.com/{username}/")
                time.sleep(3)
                
                # Try different selectors for following link
                following_link_selectors = [
                    "//a[contains(@href, '/following')]|//a[contains(text(), 'following')]|//span[contains(text(), 'following')]/parent::a",
                    "//li/a[contains(text(), 'following')]|//div[@role='button' and contains(text(), 'following')]",
                    "//div[contains(text(), 'following')]"
                ]
                
                clicked = False
                for selector in following_link_selectors:
                    try:
                        following_elements = self.driver.find_elements(By.XPATH, selector)
                        if following_elements:
                            for elem in following_elements:
                                try:
                                    elem.click()
                                    logger.info(f"Clicked following link using selector: {selector}")
                                    clicked = True
                                    time.sleep(5)  # Wait for modal to open
                                    break
                                except Exception as e:
                                    logger.debug(f"Could not click element: {e}")
                                    continue
                        if clicked:
                            break
                    except Exception as e:
                        logger.debug(f"Error with selector {selector}: {e}")
                        continue
            
            # Look for the following list - try multiple selectors
            following_list_selectors = [
                "//div[@role='dialog']//div[@role='tablist']/following-sibling::div//div[@role='button']",
                "//div[@role='dialog']//ul/li",
                "//div[contains(@class, 'following')]//ul/li",
                "//div[@role='presentation']//ul/li",
                "//div[contains(@aria-label, 'Following')]//ul/li"
            ]
            
            following_elements = []
            for selector in following_list_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements and len(elements) > 0:
                        logger.info(f"Found {len(elements)} following elements using selector: {selector}")
                        following_elements = elements[:max_count]  # Limit to max_count
                        break
                except Exception as e:
                    logger.debug(f"Error with following list selector {selector}: {e}")
                    continue
            
            if not following_elements:
                logger.warning("Could not find following elements with any selector")
                return following
            
            # Extract information from each following element
            for i, elem in enumerate(following_elements):
                try:
                    following_data = {}
                    
                    # Extract username
                    username_selectors = [
                        ".//a[contains(@href, '/')]|.//span[contains(@class, 'username')]",
                        ".//div[contains(@class, 'username')]",
                        ".//span[contains(@class, 'notranslate')]"
                    ]
                    
                    for selector in username_selectors:
                        try:
                            username_elem = elem.find_element(By.XPATH, selector)
                            username_text = username_elem.text
                            if username_text and len(username_text) > 0:
                                following_data["username"] = username_text
                                # Get profile URL
                                try:
                                    href = username_elem.get_attribute("href")
                                    if href and "instagram.com" in href:
                                        following_data["profile_url"] = href
                                    else:
                                        following_data["profile_url"] = f"https://www.instagram.com/{username_text}/"
                                except:
                                    following_data["profile_url"] = f"https://www.instagram.com/{username_text}/"
                                break
                        except:
                            continue
                    
                    # If we couldn't find username with selectors, try to extract from element text
                    if "username" not in following_data:
                        elem_text = elem.text
                        if elem_text:
                            lines = elem_text.split('\n')
                            if lines and len(lines) > 0:
                                following_data["username"] = lines[0].strip()
                                following_data["profile_url"] = f"https://www.instagram.com/{following_data['username']}/"
                    
                    # Extract profile picture
                    try:
                        img_elem = elem.find_element(By.XPATH, ".//img")
                        img_src = img_elem.get_attribute("src")
                        if img_src:
                            following_data["profile_picture_url"] = img_src
                    except:
                        following_data["profile_picture_url"] = None
                    
                    # Only add if we have at least a username
                    if "username" in following_data:
                        following.append(following_data)
                        logger.info(f"Extracted following {i+1}: {following_data['username']}")
                except Exception as e:
                    logger.debug(f"Error extracting following {i}: {e}")
            
            return following
            
        except Exception as e:
            logger.error(f"Error getting following: {e}")
            return following
    
    def get_recent_posts(self, post_count=None):
        """Get information about recent posts
        
        Args:
            post_count: Maximum number of posts to retrieve. If None, attempts to get all posts.
        """
        logger.info(f"Attempting to extract information about {'all' if post_count is None else post_count} recent posts")
        posts = []
        
        try:
            # Try different selectors for post elements
            post_selectors = [
                "//article//a[contains(@href, '/p/')]", 
                "//main//article//a[contains(@href, '/p/')]",
                "//div[contains(@class, 'post')]//a[contains(@href, '/p/')]",
                "//a[contains(@href, '/p/')]"
            ]
            
            # Set to keep track of post URLs we've already seen to avoid duplicates
            seen_post_urls = set()
            post_urls = []
            last_post_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 20  # Limit scrolling to prevent infinite loops
            
            # Scroll down to load more posts until we have enough or no more are loading
            while (post_count is None or len(seen_post_urls) < post_count) and scroll_attempts < max_scroll_attempts:
                # Find post elements with current selectors
                post_elements = []
                for selector in post_selectors:
                    try:
                        post_elements = self.driver.find_elements(By.XPATH, selector)
                        if post_elements:
                            logger.info(f"Found {len(post_elements)} posts using selector: {selector}")
                            break
                    except Exception as e:
                        logger.debug(f"Error finding posts with selector {selector}: {e}")
                        continue
                
                # Extract URLs from elements
                current_urls = []
                for elem in post_elements:
                    try:
                        url = elem.get_attribute("href")
                        if url and "/p/" in url and url not in seen_post_urls:
                            current_urls.append(url)
                            seen_post_urls.add(url)
                    except Exception as e:
                        logger.debug(f"Error extracting URL from element: {e}")
                
                # Add new URLs to our list
                post_urls.extend(current_urls)
                
                # Check if we found new posts
                if len(seen_post_urls) > last_post_count:
                    logger.info(f"Found {len(seen_post_urls)} unique posts so far")
                    last_post_count = len(seen_post_urls)
                    scroll_attempts = 0  # Reset scroll attempts if we found new posts
                else:
                    scroll_attempts += 1
                    logger.debug(f"No new posts found on scroll attempt {scroll_attempts}")
                
                # Scroll down to load more posts
                try:
                    # Scroll to the bottom of the page
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)  # Wait for posts to load
                    
                    # Try to click "Load more" button if it exists
                    load_more_selectors = [
                        "//button[contains(text(), 'Load more')]",
                        "//a[contains(text(), 'Load more')]",
                        "//div[@role='button' and contains(text(), 'Load more')]"
                    ]
                    
                    for selector in load_more_selectors:
                        try:
                            load_more_buttons = self.driver.find_elements(By.XPATH, selector)
                            if load_more_buttons:
                                for button in load_more_buttons:
                                    try:
                                        button.click()
                                        logger.info("Clicked 'Load more' button")
                                        time.sleep(3)  # Wait longer after clicking button
                                        break
                                    except:
                                        continue
                        except:
                            continue
                except Exception as e:
                    logger.debug(f"Error scrolling: {e}")
            
            if not post_urls:
                logger.warning("Could not find any posts")
                return posts
            
            logger.info(f"Found a total of {len(post_urls)} unique posts")
            
            # If post_count is specified, limit the number of posts to process
            if post_count is not None and len(post_urls) > post_count:
                post_urls = post_urls[:post_count]
                logger.info(f"Limited to processing {post_count} posts as requested")
            
            for i, post_url in enumerate(post_urls):
                logger.info(f"Processing post {i+1}/{len(post_urls)}: {post_url}")
                post_data = self.get_post_data(post_url)
                if post_data:
                    posts.append(post_data)
                    logger.info(f"Retrieved {len(post_data.get('comments', []))} comments for post {i+1}")
                else:
                    logger.warning(f"Failed to retrieve data for post {i+1}: {post_url}")
                
                # Add a slightly longer delay between posts to prevent rate limiting
                time.sleep(3)
                
            return posts
        except Exception as e:
            logger.error(f"Error extracting recent posts: {e}")
            return posts
    
    def get_post_data(self, post_url):
        """Get data for a specific post"""
        try:
            self.driver.get(post_url)
            time.sleep(3)  # Wait longer for page to load
            
            post_data = {
                "post_url": post_url,
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Save post page source for debugging
            post_id = post_url.split("/p/")[1].split("/")[0]
            with open(f"debug_post_{post_id}.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.info(f"Saved post page source to debug_post_{post_id}.html for inspection")
            
            # Get image URL - try multiple selectors with improved detection
            image_selectors = [
                "//article//img[contains(@alt, 'Photo by')]",
                "//article//img[not(@alt='')]",
                "//div[contains(@role, 'dialog')]//img",
                "//div[contains(@class, 'post')]//img",
                "//div[@role='presentation']//img[not(contains(@alt, 'profile'))]",
                "//article//div[@role='button']//img",
                "//meta[@property='og:image']" # Try to get from meta tags
            ]
            
            for selector in image_selectors:
                try:
                    img_elements = self.driver.find_elements(By.XPATH, selector)
                    if img_elements:
                        # Filter out profile pictures or small icons
                        for img in img_elements:
                            if selector == "//meta[@property='og:image']":
                                src = img.get_attribute("content")
                            else:
                                src = img.get_attribute("src")
                                
                            if src and ("scontent" in src or "/p/" in src):
                                # Check if it's not a tiny image (likely an icon)
                                if "s150x150" not in src:
                                    post_data["image_url"] = src
                                    logger.info(f"Found post image using selector: {selector}")
                                    break
                    if post_data.get("image_url"):
                        break
                except Exception as e:
                    logger.debug(f"Error finding image with selector {selector}: {e}")
                    continue
                    
            # Try to extract from meta description if still not found
            if not post_data.get("image_url"):
                try:
                    meta_image = self.driver.find_element(By.XPATH, "//meta[@property='og:image']")
                    image_url = meta_image.get_attribute("content")
                    if image_url:
                        post_data["image_url"] = image_url
                        logger.info("Found post image in meta tags")
                except Exception as e:
                    logger.debug(f"Error finding image in meta tags: {e}")
                    
            if not post_data.get("image_url"):
                logger.warning("Could not find post image")
                post_data["image_url"] = None
            
            # Get caption - first try direct caption extraction
            caption_selectors = [
                "//div[contains(@class, '_a9zs')]",
                "//div[contains(@class, 'caption')]",
                "//ul//div[contains(@role, 'button')]/span",
                "//article//h1/following-sibling::div//span",
                "//article//ul//span"
            ]
            
            caption_found = False
            for selector in caption_selectors:
                try:
                    caption_elements = self.driver.find_elements(By.XPATH, selector)
                    if caption_elements:
                        caption_text = caption_elements[0].text
                        # Check if the caption is not just a timestamp (like "19 h" or "2 d")
                        if not re.match(r'^\d+\s*[hdwmy]$', caption_text.strip()):
                            post_data["caption"] = caption_text
                            logger.info(f"Found caption using selector: {selector}")
                            caption_found = True
                            break
                except Exception as e:
                    logger.debug(f"Error finding caption with selector {selector}: {e}")
                    continue
                    
            # If no valid caption found, check if the first comment is from the post owner
            # This is often where the actual caption is stored
            if not caption_found:
                logger.info("Direct caption not found or appears to be just a timestamp. Checking first comment.")
                try:
                    # Get the username from the post URL
                    post_username = post_url.split('instagram.com/')[1].split('/')[0]
                    
                    # Try to find comments
                    comment_selectors = [
                        "//ul//li[contains(@class, 'gElp9')]",
                        "//article//ul//li//div[@role='button']/parent::div",
                        "//div[contains(text(), 'comments')]/following::ul//li",
                        "//article//ul//li[contains(@role, 'listitem')]",
                        "//ul[contains(@class, 'x78zum5') or contains(@class, '_a9ym')]//li",
                        "//article//ul//li[contains(@class, 'x1i10hfl')]",
                        "//div[contains(@class, 'xdj266r')]//ul//li",
                        "//section//ul//li[.//a[contains(@href, '/')]]",
                        "//div[@role='dialog']//ul//li",
                        "//article//ul//li[.//a]"
                    ]
                    
                    for selector in comment_selectors:
                        comment_elements = self.driver.find_elements(By.XPATH, selector)
                        if comment_elements and len(comment_elements) > 0:
                            # Check the first comment
                            first_comment = comment_elements[0]
                            comment_text = first_comment.text
                            
                            # Try to extract username from the comment
                            username_selectors = [
                                ".//a[not(contains(@href, '/explore/')) and not(contains(@role, 'link'))]",
                                ".//a[contains(@href, '/') and not(contains(@href, 'explore'))]",
                                ".//span[contains(@class, 'username')]",
                                ".//h3",
                                ".//div[contains(@class, 'username')]"
                            ]
                            
                            comment_username = None
                            for username_selector in username_selectors:
                                try:
                                    username_element = first_comment.find_element(By.XPATH, username_selector)
                                    comment_username = username_element.text
                                    if comment_username and len(comment_username) > 0:
                                        break
                                except:
                                    continue
                            
                            # If username not found with selectors, try to extract from beginning of text
                            if not comment_username:
                                username_match = re.match(r'^([\w._]+)\s', comment_text)
                                if username_match:
                                    comment_username = username_match.group(1)
                            
                            # If the first comment is from the post owner, use it as the caption
                            if comment_username and (comment_username.lower() == post_username.lower() or 
                                                    comment_username.lower() == post_username.lower().replace('@', '')):
                                # Extract the comment text without the username
                                if comment_username in comment_text:
                                    caption_text = comment_text.replace(comment_username, "", 1).strip()
                                else:
                                    caption_text = comment_text
                                
                                # Remove timestamp at the end (e.g., "19 h")
                                caption_text = re.sub(r'\s+\d+[hdwmy]$', '', caption_text)
                                
                                post_data["caption"] = caption_text
                                logger.info(f"Using first comment from post owner as caption")
                                caption_found = True
                                break
                except Exception as e:
                    logger.debug(f"Error trying to extract caption from first comment: {e}")
            
            if not caption_found:
                logger.debug("No valid caption found")
                post_data["caption"] = None
            
            # Get like count
            like_selectors = [
                "//section//span[contains(text(), 'like')]",
                "//a[contains(@href, 'liked_by')]",
                "//section//span[text() and not(contains(text(), 'and'))]"
            ]
            
            for selector in like_selectors:
                try:
                    like_elements = self.driver.find_elements(By.XPATH, selector)
                    if like_elements:
                        for elem in like_elements:
                            text = elem.text
                            if text and ("like" in text.lower() or any(c.isdigit() for c in text)):
                                post_data["likes"] = self.extract_number(text)
                                logger.info(f"Found likes using selector: {selector}")
                                break
                    if post_data.get("likes"):
                        break
                except:
                    continue
                    
            if not post_data.get("likes"):
                logger.debug("No like count found")
                post_data["likes"] = "0"
            
            # Get comments
            post_data["comments"] = self.get_post_comments()
            
            # Get timestamp
            try:
                time_elements = self.driver.find_elements(By.XPATH, "//time")
                if time_elements:
                    for time_elem in time_elements:
                        datetime_attr = time_elem.get_attribute("datetime")
                        if datetime_attr:
                            post_data["posted_at"] = datetime_attr
                            logger.info("Found post timestamp")
                            break
            except:
                post_data["posted_at"] = None
                logger.debug("No timestamp found")
                
            return post_data
            
        except Exception as e:
            logger.error(f"Error processing post {post_url}: {e}")
            return None
    
    def get_post_comments(self, max_comments=100):
        """Get comments on a post
        
        Args:
            max_comments: Maximum number of comments to retrieve. Default is 100 to get more comprehensive data.
        """
        comments = []
        
        try:
            # Try to load more comments if available (multiple attempts for better coverage)
            for _ in range(3):  # Try up to 3 times to load more comments
                try:
                    # More comprehensive selectors for load more comments buttons
                    load_more_selectors = [
                        "//span[contains(text(), 'Load more comments')]",
                        "//span[contains(text(), 'View all')]",
                        "//span[contains(text(), 'comments')]",
                        "//button[contains(text(), 'Load more comments')]",
                        "//button[contains(text(), 'View all')]",
                        "//button[contains(@aria-label, 'Load more comments')]",
                        "//div[@role='button' and contains(., 'View all')]",
                        "//div[@role='button' and contains(., 'Load more')]"
                    ]
                    
                    clicked = False
                    for selector in load_more_selectors:
                        load_more_buttons = self.driver.find_elements(By.XPATH, selector)
                        if load_more_buttons:
                            for button in load_more_buttons:
                                try:
                                    self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                                    time.sleep(1)
                                    button.click()
                                    time.sleep(3)  # Wait longer for comments to load
                                    logger.info(f"Clicked load more comments button using selector: {selector}")
                                    clicked = True
                                    break
                                except Exception as e:
                                    logger.debug(f"Error clicking button with selector {selector}: {e}")
                                    continue
                        if clicked:
                            break
                    
                    if not clicked:
                        logger.debug("No more load more comments buttons found or couldn't click them")
                        break
                except Exception as e:
                    logger.debug(f"Error trying to load more comments: {e}")
                    break
                
            # Try different selectors for comments - expanded with more modern Instagram selectors
            comment_selectors = [
                # Classic selectors
                "//ul//li[contains(@class, 'gElp9')]",
                "//article//ul//li//div[@role='button']/parent::div",
                "//div[contains(text(), 'comments')]/following::ul//li",
                "//article//ul//li[contains(@role, 'listitem')]",
                # Modern Instagram selectors
                "//ul[contains(@class, 'x78zum5') or contains(@class, '_a9ym')]//li",
                "//article//ul//li[contains(@class, 'x1i10hfl')]",
                "//div[contains(@class, 'xdj266r')]//ul//li",
                "//section//ul//li[.//a[contains(@href, '/')]]",
                "//div[@role='dialog']//ul//li",
                "//div[contains(@aria-label, 'comment')]//ul//li",
                # Generic selectors as fallback
                "//article//ul//li[.//a]",
                "//ul//li[.//a[not(contains(@href, 'explore'))]]"
            ]
            
            for selector in comment_selectors:
                try:
                    comment_elements = self.driver.find_elements(By.XPATH, selector)
                    if comment_elements:
                        logger.info(f"Found {len(comment_elements)} comments using selector: {selector}")
                        
                        # Filter out non-comment elements (like post caption)
                        filtered_comments = []
                        for elem in comment_elements:
                            # Skip elements that are likely not comments
                            if len(elem.text) < 2 or elem.text.lower() in ['like', 'reply', 'share']:
                                continue
                            filtered_comments.append(elem)
                        
                        logger.info(f"After filtering: {len(filtered_comments)} comments")
                        
                        for i, comment_elem in enumerate(filtered_comments[:max_comments]):
                            try:
                                # Try multiple selectors for username elements
                                username = None
                                username_selectors = [
                                    ".//a[not(contains(@href, '/explore/')) and not(contains(@role, 'link'))]",
                                    ".//a[contains(@href, '/') and not(contains(@href, 'explore'))]",
                                    ".//span[contains(@class, 'username')]",
                                    ".//h3",
                                    ".//div[contains(@class, 'username')]"
                                ]
                                
                                for username_selector in username_selectors:
                                    try:
                                        username_element = comment_elem.find_element(By.XPATH, username_selector)
                                        username = username_element.text
                                        if username and len(username) > 0:
                                            break
                                    except:
                                        continue
                                
                                # If we couldn't find username with selectors, try to extract from text
                                if not username:
                                    comment_text = comment_elem.text
                                    # Try to extract username from beginning of text (common pattern)
                                    username_match = re.match(r'^([\w._]+)\s', comment_text)
                                    if username_match:
                                        username = username_match.group(1)
                                    else:
                                        # Use a placeholder if we can't find the username
                                        username = "unknown_user"
                                else:
                                    comment_text = comment_elem.text
                                
                                # Get text that's not the username
                                if username in comment_text:
                                    comment_text = comment_text.replace(username, "", 1).strip()
                                
                                # Remove potential timestamp (e.g., "2d", "1w", etc.)
                                comment_text = re.sub(r'\s+\d+[dhwmy]$', '', comment_text)
                                
                                # Remove action buttons text like "Reply", "Like", etc.
                                for action in ["Reply", "Like", "See translation", "See more"]:
                                    comment_text = comment_text.replace(action, "").strip()
                                
                                # Try to extract timestamp if available
                                timestamp = None
                                time_match = re.search(r'(\d+[dhwmy])$', comment_elem.text)
                                if time_match:
                                    timestamp = time_match.group(1)
                                
                                # Only add if we have meaningful text
                                if comment_text and len(comment_text) > 0:
                                    comment_data = {
                                        "username": username,
                                        "text": comment_text
                                    }
                                    
                                    if timestamp:
                                        comment_data["timestamp"] = timestamp
                                        
                                    comments.append(comment_data)
                                    logger.debug(f"Extracted comment {i+1}: {username} - {comment_text[:30]}...")
                            except Exception as e:
                                logger.debug(f"Error parsing comment {i}: {e}")
                        
                        if comments:  # If we found comments with this selector, break the loop
                            break
                except Exception as e:
                    logger.debug(f"Error with comment selector {selector}: {e}")
                    
            if not comments:
                logger.warning("No comments found with any selector")
                    
            return comments
            
        except Exception as e:
            logger.error(f"Error extracting comments: {e}")
            return comments
    
    def save_data(self, data, username):
        """Save scraped data to JSON file"""
        if not os.path.exists("output"):
            os.makedirs("output")
            
        filename = f"output/{username}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Ensure all post data is properly formatted before saving
        if 'recent_posts' in data:
            for post in data['recent_posts']:
                # Make sure comments is always a list
                if 'comments' not in post or post['comments'] is None:
                    post['comments'] = []
                # Ensure image_url is properly set
                if 'image_url' not in post:
                    post['image_url'] = None
        
        # If this is a single post (not a profile)
        if 'post' in data:
            if 'comments' not in data['post'] or data['post']['comments'] is None:
                data['post']['comments'] = []
            if 'image_url' not in data['post']:
                data['post']['image_url'] = None
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Data saved to {filename}")
        return filename
    
    def close(self):
        """Close the webdriver"""
        if self.driver:
            self.driver.close()
            logger.info("Webdriver closed")
4

def get_instagram_account_data(username, headless=True, post_count=None):
    """
    Get data for an Instagram account without analyzing if it's fake or not.
    This function returns the raw data that would be saved to the output file.
    
    Args:
        username (str): Instagram username or profile URL
        headless (bool): Whether to run the browser in headless mode
        post_count (int): Number of recent posts to retrieve, None for all available
        
    Returns:
        dict: Raw Instagram profile data
    """
    # Format the URL if just username was provided
    if not username.startswith('http'):
        profile_url = f"https://www.instagram.com/{username}/"
    else:
        profile_url = username
    
    # Create scraper instance
    scraper = InstagramScraper(headless=headless)
    
    try:
        # Get profile information
        profile_data = scraper.get_profile_info(profile_url)
        
        # If post_count was specified, update the recent_posts with the specified count
        if post_count is not None and profile_data and 'recent_posts' in profile_data:
            profile_data['recent_posts'] = scraper.get_recent_posts(post_count)
        
        return profile_data
    except Exception as e:
        # Log error but return None to indicate failure
        print(f"Error retrieving Instagram data: {e}")
        return None
    finally:
        # Clean up resources
        scraper.close()


def main():
    parser = argparse.ArgumentParser(description='Instagram Profile Scraper')
    parser.add_argument('url', help='Instagram profile URL to scrape')
    parser.add_argument('--visible', action='store_true', help='Run in visible mode (not headless)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with extra logging')
    parser.add_argument('--post', action='store_true', help='Scrape a specific post instead of a profile')
    parser.add_argument('--post-count', type=int, help='Number of posts to scrape from profile. If not specified, attempts to scrape all posts.')
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    scraper = InstagramScraper(headless=not args.visible)
    
    try:
        if args.post:
            # Extract post ID from URL
            post_url = args.url
            post_data = scraper.get_post_data(post_url)
            
            if post_data:
                # Use post ID as filename prefix
                post_id = post_url.split("/p/")[1].split("/")[0] if "/p/" in post_url else "post"
                output_file = scraper.save_data({"post": post_data}, post_id)
                print(f"\nPost scraping completed successfully!")
                print(f"Data saved to: {output_file}")
                
                # Print summary
                print("\nPost Summary:")
                print(f"Post URL: {post_data.get('post_url')}")
                print(f"Likes: {post_data.get('likes')}")
                print(f"Comments: {len(post_data.get('comments', []))}")
                print(f"Posted at: {post_data.get('posted_at')}")
            else:
                print("Failed to retrieve post data")
        else:
            # Original profile scraping logic
            profile_data = scraper.get_profile_info(args.url)
            
            # If post_count was specified, update the recent_posts with the specified count
            if args.post_count is not None and profile_data and 'recent_posts' in profile_data:
                logger.info(f"User requested {args.post_count} posts, fetching specific number of posts")
                profile_data['recent_posts'] = scraper.get_recent_posts(args.post_count)
            
            if profile_data:
                username = profile_data["username"]
                output_file = scraper.save_data(profile_data, username)
                print(f"\nScraping completed successfully!")
                print(f"Data saved to: {output_file}")
                
                # Print summary
                print("\nProfile Summary:")
                print(f"Username: {profile_data.get('username')}")
                print(f"Full Name: {profile_data.get('full_name')}")
                print(f"Followers: {profile_data.get('followers_count')}")
                print(f"Following: {profile_data.get('following_count')}")
                print(f"Posts: {profile_data.get('posts_count')}")
                print(f"Recent Posts Retrieved: {len(profile_data.get('recent_posts', []))}")
                
                if profile_data.get("requires_login"):
                    print("\nNOTE: This profile appears to require login to view complete data.")
            else:
                print("Failed to retrieve profile data")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()