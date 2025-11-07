import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

BASE_URL = "https://science.feedback.org"

def get_article_links(pages=5):
    """Scrapre article URL from Climate Feedbacks's main feed."""
    links = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/reviews/?_topic=climate&pagination={page}"
        print(url)
        resp = requests.get(url)
        
        if resp.status_code != 200:
            print(f"Skipping page {page} (status {resp.status_code})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("h3.entry-title a"):
            links.append(a["href"])
        sleep(1)
    return links

def parse_article(url):
    """Extract article content, claim and verdict."""
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.select_one("h1,.entry-title").get_text(strip=True)
    claim = soup.find("strong", text="CLAIM:")
    if claim:
        claim = claim.find_next("p").get_text(strip=True)
    verdict = soup.find("strong", text="VERDICT:")
    if verdict:
        verdict = verdict.find_next("p").get_text(strip=True)
    return {
        "url": url,
        "title": title,
        "claim": claim,
        "verdict": verdict
    }

def scrape_climate_feedback(pages=5):
    """Scrape multiple articles into a DataFrame."""
    links = get_article_links(pages)
    data = [parse_article(link) for link in links]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = scrape_climate_feedback(pages=10)
    df.to_csv("raw/climate_feedback_articles.csv", index=False)
    print(f"Saved {len(df)} articles.")