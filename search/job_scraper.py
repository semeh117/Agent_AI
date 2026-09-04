"""
job_scraper.py
--------------
LinkedIn job scraper for Agent 2.

Responsibilities:
    1. Search LinkedIn Jobs using a query + location.
    2. Collect unique job URLs.
    3. Open each job page.
    4. Extract the job title, company, description, and available metadata.
    5. Return standardized job dictionaries.

This module does NOT:
    - extract skills with an LLM
    - calculate embeddings
    - calculate cosine similarity
    - rank jobs
    - generate cover letters
    - save CSV files

The production wrapper and ranking orchestration live in:
    agent/tools/linkedin_match_tool.py
    pipeline/linkedin_cosine_pipeline.py

Public function:
    search_jobs(query, location="", max_jobs=20)
"""

import time
import re
import json
import logging
from typing import Optional
from urllib.parse import quote_plus

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_JOBS = 20

SCROLL_ROUNDS = 15
PAGE_LOAD_DELAY = 4
SCROLL_DELAY = 1.5
JOB_PAGE_DELAY = 3

# Your current Chrome version
CHROME_VERSION = 151


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _create_driver():
    """
    Create the Selenium Chrome driver used for LinkedIn scraping.
    """

    options = uc.ChromeOptions()

    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(
        options=options,
        version_main=CHROME_VERSION
    )

    return driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text_from_selectors(driver, selectors) -> str:
    """
    Try several CSS selectors and return the first non-empty text.
    """

    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)

            for element in elements:
                text = element.text.strip()

                if text:
                    return text

        except Exception:
            continue

    return ""


def _build_search_url(
    query: str,
    location: str = "",
    posted_within_hours: Optional[int] = None,
) -> str:
    """
    Build a LinkedIn public jobs search URL.

    Example:
        query="Machine Learning Engineer"
        location="Ireland"

    -> LinkedIn jobs search URL
    """

    parameters = [f"keywords={quote_plus(query.strip())}"]
    if location:
        parameters.append(f"location={quote_plus(location.strip())}")
    if posted_within_hours is not None:
        if posted_within_hours <= 0:
            raise ValueError("posted_within_hours must be greater than zero.")
        parameters.append(f"f_TPR=r{int(posted_within_hours) * 3600}")
    return "https://www.linkedin.com/jobs/search/?" + "&".join(parameters)


def _job_url_identity(url: str) -> str:
    """Normalize locale/tracking variants to one LinkedIn job identity."""

    clean = str(url or "").split("?", 1)[0].rstrip("/")
    match = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)$", clean)
    return f"linkedin:{match.group(1)}" if match else clean.casefold()


# ---------------------------------------------------------------------------
# Collect job URLs
# ---------------------------------------------------------------------------

def _collect_job_urls(
    driver,
    query: str,
    location: str,
    max_jobs: int,
    posted_within_hours: Optional[int] = None,
    exclude_urls: Optional[set[str]] = None,
) -> list[str]:
    """
    Search LinkedIn and collect unique job URLs.

    This function only collects URLs.
    It does NOT scrape the full job descriptions yet.
    """

    search_url = _build_search_url(query, location, posted_within_hours)

    logger.info("Opening LinkedIn search:")
    logger.info(search_url)

    driver.get(search_url)

    time.sleep(PAGE_LOAD_DELAY)

    job_urls: list[str] = []
    seen_urls: set[str] = set()
    excluded_urls = {
        _job_url_identity(url)
        for url in (exclude_urls or set())
    }

    no_new_url_rounds = 0

    for round_number in range(SCROLL_ROUNDS):

        # ---------------------------------------------------------------
        # Collect all currently visible job links
        # ---------------------------------------------------------------

        try:
            links = driver.find_elements(
                By.CSS_SELECTOR,
                'a[href*="/jobs/view/"]'
            )

            before = len(job_urls)

            for link in links:

                try:
                    href = link.get_attribute("href")

                    if not href:
                        continue

                    # Remove tracking/query parameters
                    href = href.split("?")[0]

                    if "/jobs/view/" not in href:
                        continue

                    href = href.rstrip("/")
                    identity = _job_url_identity(href)
                    if identity in excluded_urls or identity in seen_urls:
                        continue

                    seen_urls.add(identity)
                    job_urls.append(href)

                    if len(job_urls) >= max_jobs:
                        break

                except Exception:
                    continue

            added = len(job_urls) - before

            logger.info(
                "Scroll %s/%s | found %s job URLs (+%s)",
                round_number + 1,
                SCROLL_ROUNDS,
                len(job_urls),
                added
            )

            if added == 0:
                no_new_url_rounds += 1
            else:
                no_new_url_rounds = 0

        except Exception as exc:
            logger.warning("Could not collect job URLs: %s", exc)

        # Enough jobs
        if len(job_urls) >= max_jobs:
            break

        # ---------------------------------------------------------------
        # Try to click "See more jobs" / "Show more"
        # ---------------------------------------------------------------

        try:

            buttons = driver.find_elements(
                By.XPATH,
                """
                //button[
                    contains(
                        translate(
                            normalize-space(.),
                            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                            'abcdefghijklmnopqrstuvwxyz'
                        ),
                        'see more'
                    )
                    or
                    contains(
                        translate(
                            normalize-space(.),
                            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                            'abcdefghijklmnopqrstuvwxyz'
                        ),
                        'show more'
                    )
                ]
                """
            )

            for button in buttons:

                try:
                    if button.is_displayed() and button.is_enabled():
                        driver.execute_script(
                            "arguments[0].click();",
                            button
                        )

                        time.sleep(1)

                        break

                except Exception:
                    continue

        except Exception:
            pass

        # ---------------------------------------------------------------
        # Scroll down
        # ---------------------------------------------------------------

        try:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
        except Exception:
            pass

        time.sleep(SCROLL_DELAY)

        # If LinkedIn stopped giving us anything new for several rounds,
        # stop instead of endlessly scrolling.
        if no_new_url_rounds >= 3:
            logger.info(
                "No new job URLs found for several rounds. Stopping search."
            )
            break

    return job_urls[:max_jobs]


# ---------------------------------------------------------------------------
# Scrape one job page
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """
    Normalize scraped LinkedIn text.
    """

    if not text:
        return ""

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def _scrape_criteria(driver) -> tuple:
    """
    Collect the job-criteria chips from the LinkedIn job card
    ("Seniority level", "Employment type", ...) and classify them into the
    project's seniority list + employment_type string by keyword. Wrapped in
    try/except everywhere — LinkedIn's markup changes often and a missing
    chip must never fail the scrape.
    """
    chip_texts = []

    selectors = [
        ".jobs-unified-top-card__job-insight",
        ".job-criteria__text",
        "ul.jobs-description__footer li",
    ]

    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)

            for element in elements:
                text = element.text.strip()

                if text:
                    chip_texts.append(text)
        except Exception:
            continue

    seniority = []
    employment_types = []

    for text in chip_texts:
        low = text.lower()

        if re.search(
            r"full-time|part-time|contract|temporary|internship|freelance",
            low,
        ):
            employment_types.append(text)

        if re.search(
            r"\b(entry level|associate|mid-senior level|senior|director|executive|intern|junior)\b",
            low,
        ):
            seniority.append(text)

    return seniority, employment_types


def _click_see_more(driver) -> None:
    """
    LinkedIn collapses long job descriptions behind a "See more" button.
    Click it (when present) so the FULL description text is available,
    instead of scraping only the preview.
    """
    try:
        buttons = driver.find_elements(
            By.XPATH,
            """
            //button[
                contains(
                    translate(
                        normalize-space(.),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        'abcdefghijklmnopqrstuvwxyz'
                    ),
                    'see more'
                )
                or
                contains(
                    translate(
                        normalize-space(.),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        'abcdefghijklmnopqrstuvwxyz'
                    ),
                    'show more'
                )
            ]
            """
        )

        for button in buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script("arguments[0].click();", button)
                    time.sleep(1)
                    break
            except Exception:
                continue
    except Exception:
        pass

def _scrape_description(driver) -> str:
    """
    Best-effort FULL job description. After _click_see_more(), the complete
    posting lives inside .show-more-less-html__markup. Prefer that
    container's innerText (JS innerText loads everything Selenium's .text
    sometimes misses), and among all matches keep the LONGEST one — the
    first match is often a short preview copied elsewhere on the page.
    """
    container_selectors = [
        ".jobs-description__content .show-more-less-html__markup",
        ".show-more-less-html__markup",
        ".jobs-description__content",
        ".jobs-box__html-content",
        ".description__text",
    ]

    best = ""

    for selector in container_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
        except Exception:
            continue

        for element in elements:
            try:
                text = driver.execute_script(
                    "return arguments[0].innerText;", element
                )
            except Exception:
                text = getattr(element, "text", "") or ""

            text = (text or "").strip()

            # Ignore tiny/unrelated elements (chips, buttons, footers).
            if len(text) >= 50 and len(text) > len(best):
                best = text

    return _clean_text(best)


def _walk_json(value):
    """Yield every dictionary contained in a JSON-compatible value."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _scrape_structured_job_data(driver) -> dict:
    """Read title/company from LinkedIn's schema.org JobPosting JSON-LD.

    Public job pages commonly expose structured metadata for search engines.
    It is less sensitive to visible-page localization and CSS changes than
    scraping a generic heading such as ``h1``. All failures are best-effort:
    callers still have selector and browser-tab fallbacks.
    """

    try:
        scripts = driver.find_elements(
            By.CSS_SELECTOR,
            'script[type="application/ld+json"]',
        )
    except Exception:
        return {}

    for script in scripts:
        try:
            raw = script.get_attribute("textContent") or script.get_attribute(
                "innerHTML"
            )
            data = json.loads(raw or "")
        except Exception:
            continue

        for item in _walk_json(data):
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if "JobPosting" not in types:
                continue

            organization = item.get("hiringOrganization") or {}
            if isinstance(organization, list):
                organization = next(
                    (entry for entry in organization if isinstance(entry, dict)),
                    {},
                )

            return {
                "title": str(item.get("title") or "").strip(),
                "company": str(
                    (organization.get("name") or "")
                    if isinstance(organization, dict)
                    else ""
                ).strip(),
            }

    return {}


def _strip_parenthesized_location(text: str) -> str:
    """Remove a final ``(City, Region, Country)`` without removing ``(f/m/d)``."""

    before, separator, suffix = text.rpartition(" (")
    if separator and suffix.endswith(")") and "," in suffix:
        return before.strip()
    return text.strip()


def _parse_browser_tab_title(tab_title: str) -> tuple[str, str]:
    """Parse role/company from common localized LinkedIn tab-title forms."""

    text = (tab_title or "").split(" | LinkedIn")[0].strip()
    if not text:
        return "", ""

    # French example observed on de.linkedin.com with a French browser UI:
    # "adjoe recrute pour des postes de Senior ML Engineer (Hambourg, ...)"
    french_marker = " recrute pour des postes de "
    if french_marker in text:
        company, title = text.split(french_marker, 1)
        return _strip_parenthesized_location(title), company.strip()

    # Public English pages commonly use "Company hiring Role in Location".
    if " hiring " in text:
        company, title_and_location = text.split(" hiring ", 1)
        title, separator, _location = title_and_location.rpartition(" in ")
        return (title if separator else title_and_location).strip(), company.strip()

    # German public-page fallback: "Company sucht Role in Location".
    if " sucht " in text:
        company, title_and_location = text.split(" sucht ", 1)
        title, separator, _location = title_and_location.rpartition(" in ")
        return (title if separator else title_and_location).strip(), company.strip()

    if " at " in text:
        role, _, company = text.rpartition(" at ")
        return role.strip(), company.strip()

    return text, ""


def _browser_tab_parts(driver) -> tuple:
    """
    Fallback source for title/company: LinkedIn's browser tab title is
    'Title at Company | LinkedIn' (or 'Company hiring Title in Location |
    LinkedIn'). Used only when the in-page selectors come up empty.
    """
    tab_title = ""

    try:
        tab_title = driver.title or ""
    except Exception:
        tab_title = ""

    return _parse_browser_tab_title(tab_title)


def _scrape_job_page(driver, url: str) -> dict:
    """
    Open one LinkedIn job page and extract job information.

    Only uses selectors that are intended to represent the job
    description. We deliberately avoid generic large <div> fallbacks
    because LinkedIn pages contain lots of unrelated job-card/sidebar
    text.
    """

    job = {
        "title": "",
        "company": "",
        "employment_type": None,
        "seniority": [],
        "description": "",
        "categories": [],
        "requirements": [],
        "skills": [],
        "url": url,
        "salary_min": None,
        "salary_max": None,
    }

    try:
        driver.get(url)

        time.sleep(JOB_PAGE_DELAY)

        # Expand the full description if it is collapsed.
        _click_see_more(driver)

        # Prefer schema.org metadata for identity fields. Unlike visible
        # headings and browser titles, these values are normally not wrapped
        # in localized phrases such as "recrute pour des postes de".
        structured_job = _scrape_structured_job_data(driver)

        # ---------------------------------------------------------------
        # Seniority / employment type chips
        # ---------------------------------------------------------------

        job["seniority"], employment_types = _scrape_criteria(driver)

        if employment_types:
            job["employment_type"] = employment_types[0]

        # ---------------------------------------------------------------
        # Description — full text for downstream requirement extraction.
        # The LLM needs the whole posting, so re-read a few times and keep
        # the longest version: the description can keep growing after the
        # "See more" click + lazy loading.
        # ---------------------------------------------------------------

        description = ""

        for _ in range(6):
            candidate = _scrape_description(driver)

            if len(candidate) > len(description):
                description = candidate

            # Good enough — a genuine full posting is rarely under this.
            if len(description) >= 300:
                break

            time.sleep(1)

        job["description"] = description

        # ---------------------------------------------------------------
        # Title / company — read AFTER the description poll above, by which
        # point the top card is fully rendered. Try the current and legacy
        # LinkedIn selectors, then fall back to the browser tab title.
        # ---------------------------------------------------------------

        job["title"] = _get_text_from_selectors(
            driver,
            [
                "h1.job-details-jobs-unified-top-card__job-title",
                "h1.top-card-layout__title",
                "h1.topcard__title",
                "h1",
            ]
        )

        job["company"] = _get_text_from_selectors(
            driver,
            [
                "a.job-details-jobs-unified-top-card__company-name",
                ".job-details-jobs-unified-top-card__company-name",
                "a.topcard__org-name-link",
                ".topcard__org-name-link",
                "a[data-tracking-control-name='public_jobs_topcard-org-name']",
            ]
        )

        # Structured data is the most precise source when present. Override
        # generic selector results, which can accidentally capture a complete
        # localized page heading instead of the actual role/company fields.
        if structured_job.get("title"):
            job["title"] = structured_job["title"]
        if structured_job.get("company"):
            job["company"] = structured_job["company"]

        if not job["title"] or not job["company"]:
            tab_role, tab_company = _browser_tab_parts(driver)

            if not job["title"]:
                job["title"] = tab_role
            if not job["company"]:
                job["company"] = tab_company

        # ---------------------------------------------------------------
        # Clean values
        # ---------------------------------------------------------------
        
        job["title"] = _clean_text(job["title"])
        job["company"] = _clean_text(job["company"])
        # Preserve paragraph and bullet boundaries for the requirement parser.
        # Flattening the description into one line can make an ``including``
        # or ``such as`` clause absorb requirements from later bullets.
        job["description"] = _clean_text(job["description"])
        return job

    except Exception as exc:

        logger.warning(
            "Failed to scrape job %s: %s",
            url,
            exc
        )

        return job
# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

def search_jobs(
    query: str,
    location: str = "",
    max_jobs: int = DEFAULT_MAX_JOBS,
    posted_within_hours: Optional[int] = None,
    exclude_urls: Optional[set[str]] = None,
) -> list[dict]:
    """
    Search LinkedIn and return standardized job postings.

    Parameters
    ----------
    query:
        Search query generated by Agent 2.

        Example:
            "Machine Learning Engineer Python PyTorch"

    location:
        Location used for LinkedIn search.

        Example:
            "Ireland"

    max_jobs:
        Maximum number of jobs to return.

    Returns
    -------
    list[dict]
        Standardized job postings.

    Example
    -------
    jobs = search_jobs(
        query="Machine Learning Engineer Python PyTorch",
        location="Ireland",
        max_jobs=10
    )
    """

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    query = (query or "").strip()
    location = (location or "").strip()

    if not query:
        raise ValueError("LinkedIn search query cannot be empty.")

    if max_jobs <= 0:
        raise ValueError("max_jobs must be greater than 0.")

    max_jobs = min(max_jobs, 50)
    if posted_within_hours is not None and posted_within_hours <= 0:
        raise ValueError("posted_within_hours must be greater than zero.")

    driver = None

    try:

        # -----------------------------------------------------------
        # Create browser
        # -----------------------------------------------------------

        driver = _create_driver()

        logger.info(
            "Searching LinkedIn for '%s' in '%s'",
            query,
            location or "any location"
        )

        # -----------------------------------------------------------
        # Step 1:
        # Collect job URLs
        # -----------------------------------------------------------

        urls = _collect_job_urls(
            driver=driver,
            query=query,
            location=location,
            max_jobs=max_jobs,
            posted_within_hours=posted_within_hours,
            exclude_urls=exclude_urls,
        )

        logger.info(
            "Collected %s unique LinkedIn job URLs.",
            len(urls)
        )

        if not urls:
            logger.warning(
                "No LinkedIn jobs found for query='%s', location='%s'",
                query,
                location
            )

            return []

        # -----------------------------------------------------------
        # Step 2:
        # Open every job URL and scrape full descriptions
        # -----------------------------------------------------------

        jobs = []

        for index, url in enumerate(urls, start=1):

            logger.info(
                "Scraping job %s/%s",
                index,
                len(urls)
            )

            job = _scrape_job_page(
                driver=driver,
                url=url
            )

            # Only keep jobs where we actually obtained useful data.
            if job["title"] or job["description"]:

                jobs.append(job)

            # Small delay between job pages
            time.sleep(JOB_PAGE_DELAY)

        logger.info(
            "Successfully scraped %s/%s LinkedIn jobs.",
            len(jobs),
            len(urls)
        )

        return jobs

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
            # undetected_chromedriver's Chrome.__del__ calls quit() a second
            # time when the object is garbage-collected (or at interpreter
            # shutdown), re-killing the already-dead Chrome process and
            # raising a noisy `OSError: [WinError 6]` on Windows. Replace it
            # with a no-op so teardown stays a single clean quit.
            try:
                driver.quit = lambda: None
            except Exception:
                pass
        driver = None
