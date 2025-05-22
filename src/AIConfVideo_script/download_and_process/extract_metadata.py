#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_driver():
    """Set up and return a headless Chrome webdriver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1920,1080")

    # Add a realistic user agent
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    chrome_options.add_argument(f"--user-agent={user_agent}")

    driver = webdriver.Chrome(options=chrome_options)

    # Prevent webdriver detection
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    return driver


def find_slideslive_id(content, patterns):
    """Extract SlidesLive ID using regex patterns"""
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None


def extract_slideslive_openreview_id(site_url, quiet=False):
    """Extract the SlidesLive ID and OpenReview ID from the NeurIPS page"""
    if not quiet:
        print("Extracting SlidesLive and OpenReview IDs from URL... \n", end="", flush=True)

    # Patterns to match SlidesLive ID
    patterns = [r"embed/(\d+)", r"slideslive\.com/(\d+)", r"presentation/(\d+)"]

    # Patterns to match OpenReview ID
    openreview_patterns = [r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)"]

    driver = setup_driver()
    slideslive_id = None
    openreview_id = None

    try:
        driver.get(site_url)
        time.sleep(5)  # Wait for page to load

        # First try to find ID in iframes
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe")))
            iframes = driver.find_elements(By.CSS_SELECTOR, "iframe")

            for iframe in iframes:
                iframe_src = iframe.get_attribute("src")
                slideslive_id = find_slideslive_id(iframe_src, patterns)
                if slideslive_id:
                    break
        except Exception as e:
            if not quiet:
                logger.warning(f"No iframes found or error: {e}")

        # If not found in iframes, try page source
        if not slideslive_id:
            page_source = driver.page_source
            slideslive_id = find_slideslive_id(page_source, patterns)

        # Look for OpenReview ID in specific link elements
        try:
            links = driver.find_elements(By.CSS_SELECTOR, "a")
            for link in links:
                href = link.get_attribute("href")
                if href and "openreview.net" in href:
                    for pattern in openreview_patterns:
                        match = re.search(pattern, href)
                        if match:
                            openreview_id = match.group(1)
                            break
                    if openreview_id:
                        break
        except Exception as e:
            if not quiet:
                logger.warning(f"Error finding OpenReview ID: {e}")

        if not openreview_id:
            # Try to find in page source if not found in links
            page_source = driver.page_source if "page_source" in locals() else driver.page_source
            for pattern in openreview_patterns:
                match = re.search(pattern, page_source)
                if match:
                    openreview_id = match.group(1)
                    break

        return slideslive_id, openreview_id

    except Exception as e:
        if not quiet:
            logger.error(f"Error extracting SlidesLive ID: {e}")
        return None, None
    finally:
        driver.quit()


def format_seconds_to_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"


def get_slide_timestamp(driver, total_duration_seconds):
    """Extract timestamp from progress bar"""
    try:
        progress_slide = driver.find_element(By.CSS_SELECTOR, "[data-slp-target='progressbarSlide']")
        style = progress_slide.get_attribute("style")

        match = re.search(r"left:\s*([\d.]+)%", style)
        if match:
            percentage = float(match.group(1)) / 100
            timestamp_seconds = total_duration_seconds * percentage
            timestamp = format_seconds_to_timestamp(timestamp_seconds)
            return timestamp, timestamp_seconds, percentage * 100
    except Exception as e:
        logger.warning(f"Error getting progress bar percentage: {e}")

    return "0:00", 0, 0


def get_slide_image_url(driver):
    """Get the current slide image URL"""
    try:
        selectors = ["img.vp-slide-image", "img[alt='Image slide']"]
        for selector in selectors:
            try:
                slide_img = driver.find_element(By.CSS_SELECTOR, selector)
                return slide_img.get_attribute("src")
            except:
                continue
    except Exception as e:
        logger.warning(f"Error getting slide image URL: {e}")
    return None


def extract_from_slideslive(slideslive_id, slides_dir="data", quiet=False):
    """Extract slide metadata (timestamps) without downloading the actual slide images"""
    if not quiet:
        print(f"Extracting slide metadata from SlidesLive ID: {slideslive_id}")

    slides_url = f"https://slideslive.com/{slideslive_id}"
    # slides_dir = "data"
    os.makedirs(slides_dir, exist_ok=True)

    driver = setup_driver()
    slide_data = []
    total_slides = 100  # Default value
    total_duration_seconds = 0

    try:
        if not quiet:
            print("Navigating to SlidesLive page...")
        driver.get(slides_url)
        time.sleep(5)  # Wait for page to load

        # Get total slides count
        try:
            slide_count_element = driver.find_element(By.CSS_SELECTOR, "[data-slp-target='slideCount']")
            total_slides = int(slide_count_element.text)
        except Exception as e:
            logger.warning(f"Using default slide count: {e}")

        # Get video duration
        try:
            duration_element = driver.find_element(By.CSS_SELECTOR, "[data-slp-target='duration']")
            duration_text = duration_element.text

            duration_parts = duration_text.split(":")
            if len(duration_parts) == 2:
                total_duration_seconds = int(duration_parts[0]) * 60 + int(duration_parts[1])
            elif len(duration_parts) == 3:
                total_duration_seconds = (
                    int(duration_parts[0]) * 3600 + int(duration_parts[1]) * 60 + int(duration_parts[2])
                )
        except Exception as e:
            logger.warning(f"Error getting video duration: {e}")

        # Wait for slide images to be visible
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img.vp-slide-image, img[alt='Image slide']"))
        )

        # Get timestamp for first slide
        timestamp, timestamp_seconds, percentage = get_slide_timestamp(driver, total_duration_seconds)
        img_url = get_slide_image_url(driver)

        # Add first slide data
        slide_data.append(
            {
                "slide_number": 1,
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
                "percentage": percentage,
                "image_url": img_url,
            }
        )

        # Set up navigation method (button or keyboard)
        next_button_selector = "div.slp__bigButton.slp__bigButton--next[data-slp-action='nextSlide']"
        use_keyboard_nav = False
        next_button = None

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, next_button_selector))
            )
        except Exception:
            use_keyboard_nav = True
            # Click on the slide area to ensure focus for keyboard navigation
            try:
                slide_area = driver.find_element(By.CSS_SELECTOR, ".slp__slideContainer, .vp-slides-wrapper")
                slide_area.click()
            except:
                # Click in the middle of the page
                actions = ActionChains(driver)
                actions.move_to_element_with_offset(driver.find_element(By.TAG_NAME, "body"), 500, 500)
                actions.click().perform()

        # Process remaining slides
        current_slide = 1  # We already processed slide 1

        # Create a progress bar
        if not quiet:
            print(f"Extracting timing data for {total_slides} slides:")
        progress_bar = tqdm(total=total_slides, unit="slide", disable=quiet)
        progress_bar.update(1)  # Update for the first slide we already processed

        while current_slide < total_slides:
            try:
                # Navigate to next slide
                if use_keyboard_nav:
                    ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
                else:
                    driver.execute_script("arguments[0].click();", next_button)

                time.sleep(1)  # Wait for slide to load

                # Get current slide number
                try:
                    current_slide_element = driver.find_element(By.CSS_SELECTOR, "[data-slp-target='currentSlide']")
                    displayed_slide = int(current_slide_element.text)
                except:
                    displayed_slide = current_slide + 1

                # Update progress bar for any skipped slides
                if displayed_slide > current_slide + 1:
                    progress_bar.update(displayed_slide - current_slide - 1)

                current_slide = displayed_slide

                # Get slide data
                timestamp, timestamp_seconds, percentage = get_slide_timestamp(driver, total_duration_seconds)
                img_url = get_slide_image_url(driver)

                # Add slide data
                slide_data.append(
                    {
                        "slide_number": current_slide,
                        "timestamp": timestamp,
                        "timestamp_seconds": timestamp_seconds,
                        "percentage": percentage,
                        "image_url": img_url,
                    }
                )

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_description(f"Slide {current_slide}/{total_slides}")

            except Exception as e:
                logger.error(f"Error processing slide {current_slide + 1}: {e}")

                # Try to recover navigation
                if not use_keyboard_nav:
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, next_button_selector)
                    except:
                        logger.warning("Switching to keyboard navigation")
                        use_keyboard_nav = True

        # Close the progress bar
        progress_bar.close()

    except Exception as e:
        logger.error(f"Error extracting slide metadata: {e}")
    finally:
        # Save the slide data to a JSON file
        output_file = f"{slides_dir}/slideslive_{slideslive_id}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "presentation_id": slideslive_id,
                    "slideslive_url": slides_url,
                    "total_slides": total_slides,
                    "total_duration": total_duration_seconds,
                    "total_duration_formatted": format_seconds_to_timestamp(total_duration_seconds),
                    "slides": slide_data,
                },
                f,
                indent=2,
            )

        if not quiet:
            print(f"Extracted timing data for {len(slide_data)} of {total_slides} slides to {output_file}")
        driver.quit()

    return output_file


def extract_from_openreview(openreview_id, quiet=False):
    """
    Extract the TLDR and abstract from the Paper Site:
    1. Go to the Paper Site
    2. Click on the "OpenReview" link and go to the OpenReview site
    3. Find the "TLDR" and "Abstract" fields and extract the text
    4. Return the TLDR and abstract, return None if either is not found
    """
    openreview_url = f"https://openreview.net/forum?id={openreview_id}"

    if not quiet:
        print(f"Extracting TLDR and abstract from OpenReview: {openreview_id}")

    driver = setup_driver()
    tldr = None
    abstract = None

    try:
        driver.get(openreview_url)
        time.sleep(5)  # Wait for page to load

        # Look for TLDR - it's typically in a field with a label "TL;DR" or "TLDR"
        try:
            # First try to find TLDR using the exact structure provided
            tldr_elements = driver.find_elements(
                By.XPATH,
                "//strong[contains(@class, 'note-content-field') and contains(text(), 'TL;DR')]/following::span[contains(@class, 'note-content-value')][1]",
            )

            if tldr_elements:
                tldr = tldr_elements[0].text.strip()
            else:
                # Try alternative xpath for similar structures
                tldr_elements = driver.find_elements(
                    By.XPATH, "//strong[contains(text(), 'TL;DR') or contains(text(), 'TLDR')]/following::span[1]"
                )
                if tldr_elements:
                    tldr = tldr_elements[0].text.strip()
                else:
                    # Original methods as fallback
                    tldr_elements = driver.find_elements(
                        By.XPATH, "//h4[contains(text(), 'TL;DR')]/following-sibling::div"
                    )
                    if tldr_elements:
                        tldr = tldr_elements[0].text.strip()
                    else:
                        tldr_elements = driver.find_elements(
                            By.XPATH, "//strong[contains(text(), 'TL;DR')]/following::text()[1]"
                        )
                        if tldr_elements:
                            tldr = tldr_elements[0].strip()
        except Exception as e:
            if not quiet:
                logger.warning(f"Error finding TLDR: {e}")

        # Look for Abstract
        try:
            # Try to find abstract by looking for "Abstract" heading or label
            abstract_elements = driver.find_elements(
                By.XPATH, "//h4[contains(text(), 'Abstract')]/following-sibling::div"
            )
            if abstract_elements:
                abstract = abstract_elements[0].text.strip()
            else:
                # Alternative method - sometimes abstract is in a note content section
                abstract_elements = driver.find_elements(By.CSS_SELECTOR, ".note-content-value")
                for element in abstract_elements:
                    heading = element.find_element(By.XPATH, "./preceding-sibling::*[1]").text
                    if "abstract" in heading.lower():
                        abstract = element.text.strip()
                        break
        except Exception as e:
            if not quiet:
                logger.warning(f"Error finding abstract: {e}")

        if not quiet:
            if tldr:
                print(f"📌 Found TLDR ({len(tldr)} chars)")
            else:
                print("❌ No TLDR found")

            if abstract:
                print(f"📄 Found abstract ({len(abstract)} chars)")
            else:
                print("❌ No abstract found")

        return tldr, abstract

    except Exception as e:
        if not quiet:
            logger.error(f"Error extracting data from OpenReview: {e}")
        return None, None
    finally:
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract slide metadata from SlidesLive presentations embedded in NeurIPS pages"
    )
    parser.add_argument("url", help="NeurIPS URL containing a SlidesLive presentation")
    parser.add_argument(
        "--id-only",
        action="store_true",
        help="Only extract and print the SlidesLive ID without processing metadata",
    )
    args = parser.parse_args()

    # Extract SlidesLive ID and OpenReview ID from the provided URL
    slideslive_id, openreview_id = extract_slideslive_openreview_id(args.url)

    if slideslive_id:
        print(f"🎬 Found SlidesLive ID: {slideslive_id}")
        if openreview_id:
            print(f"📝 Found OpenReview ID: {openreview_id}")
        else:
            print("❌ No OpenReview ID found")

        if not args.id_only:
            output_file = extract_from_slideslive(slideslive_id, slides_dir="data", quiet=False)
            print(f"💾 Slide metadata saved to: {output_file}")

            tldr, abstract = extract_from_openreview(openreview_id, quiet=False)
    else:
        print("❌ Failed to extract SlidesLive ID")
        exit(1)

    print("✅ Process completed successfully!")
    exit(0)
