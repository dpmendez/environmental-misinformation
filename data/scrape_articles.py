import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://science.feedback.org/reviews"
TOPIC = "climate"

def scrape_article_page(link):
    """Fetch claim and verdict from the article page if missing."""
    if not link:
        return None, None
    if link.startswith("/"):
        link = "https://science.feedback.org" + link
    resp = requests.get(link)
    if resp.status_code != 200:
        return None, None
    soup = BeautifulSoup(resp.text, "html.parser")
    claim_tag = soup.select_one(".reviewed-content__quote p")
    verdict_tag = soup.select_one(".story__label")
    claim = claim_tag.get_text(strip=True) if claim_tag else None
    verdict = verdict_tag.get_text(strip=True) if verdict_tag else None
    return claim, verdict

def scrape_climate_feedback(pages=5, delay=2):
    """Scrape multiple Climate Feedback review pages."""
    data = []

    for page in range(1, pages + 1):
        if page == 1:
            url = f"{BASE_URL}/?_topic={TOPIC}"
        else:
            url = f"{BASE_URL}/?_topic={TOPIC}&_pagination={page}"

        print(f"Fetching: {url}")
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Skipping page {page} (status {resp.status_code})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article.story")

        if not articles:
            print(f"No articles found on page {page}")
            continue

        for art in articles:
            title_tag = art.select_one("h2.story__title a")
            title = title_tag.get_text(strip=True) if title_tag else None
            link = title_tag["href"] if title_tag else None
            if link and link.startswith("/"):
                link = "https://science.feedback.org" + link

            claim_tag = art.select_one(".reviewed-content__quote p")
            claim = claim_tag.get_text(strip=True) if claim_tag else None

            verdict_tag = art.select_one(".story__label")
            verdict = verdict_tag.get_text(strip=True) if verdict_tag else None

            date_tag = art.select_one(".story__posted-on")
            date = date_tag.text.strip().replace("Posted on:", "").strip() if date_tag else None

            # If missing claim/verdict, fetch from article page
            if not claim or not verdict:
                article_claim, article_verdict = scrape_article_page(link)
                claim = claim or article_claim
                verdict = verdict or article_verdict

            data.append({
                "title": title,
                "claim": claim,
                "verdict": verdict,
                "url": link,
                "date": date
            })

        print(f"Collected {len(articles)} articles from page {page}")
        time.sleep(delay)

    df = pd.DataFrame(data)
    df.to_csv("raw/climate_feedback_articles.csv", index=False)
    print(f"\nSaved {len(df)} total articles to raw/climate_feedback_articles.csv")

    return df

if __name__ == "__main__":
    df = scrape_climate_feedback(pages=5)
    print(df.head())