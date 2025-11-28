import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

base_url = "https://www.tayara.tn"
search_url_template = "https://www.tayara.tn/ar/listing/c/v%C3%A9hicules/voitures/?minPrice=8000&maxPrice=300000&page={}"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MAX_PAGES = 149       
MAX_WORKERS = 12      
DETAIL_SLEEP = 0.05    

all_car_data = []
all_columns = set()

def make_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(500,502,503,504))
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))
    s.headers.update(headers)
    return s

def clean_price(text):
    if not text:
        return ""
    t = re.sub(r"[^\d]", "", text)
    return t

def collect_item_urls(session, pages=MAX_PAGES):
    urls = []
    for page in range(1, pages + 1):
        try:
            print(f"Listing page {page}")
            resp = session.get(search_url_template.format(page), timeout=15)
            if resp.status_code != 200:
                print(f"Failed listing {page}: {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = soup.find_all("article")
            if not articles:
                print("No articles found, stopping listings")
                break
            for article in articles:
                a = article.find("a", href=True)
                if not a:
                    continue
                href = a["href"]
                if href.startswith("/ar/item/"):
                    full = base_url + href
                    urls.append(full)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error collecting page {page}: {e}")
            continue
    # keep order but deduplicate
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def parse_detail(url, session):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"Detail fetch failed {url}: {resp.status_code}")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        car_info = {"URL": url}

        price_tag = soup.find("data", class_=lambda c: c and "font-arabic" in c)
        if not price_tag:
            candidates = soup.find_all(text=re.compile(r"\bDT\b", re.I))
            price_text = candidates[0] if candidates else ""
        else:
            price_text = price_tag.get_text(" ", strip=True)
        price_clean = clean_price(price_text)
        if price_clean:
            car_info["Price"] = price_clean
        specs_ul = None
        for ul in soup.find_all("ul"):
            classes = ul.get("class") or []
            cls = " ".join(classes)
            if "grid" in cls and "grid-cols-12" in cls:
                specs_ul = ul
                break

        has_marque = False
        has_modele = False
        if specs_ul:
            items = specs_ul.find_all("li")
            for item in items:
                flex_span = item.find("span", class_=lambda c: c and "flex" in c and "flex-col" in c)
                if not flex_span:
                    spans = item.find_all("span", recursive=False)
                    if spans:
                        for s in spans:
                            inner = s.find_all("span", recursive=False)
                            if len(inner) >= 2:
                                flex_span = s
                                break
                if not flex_span:
                    continue
                inner_spans = flex_span.find_all("span", recursive=False)
                if len(inner_spans) < 2:
                    continue
                label = inner_spans[0].get_text(" ", strip=True)
                value = inner_spans[1].get_text(" ", strip=True)
                if not label:
                    continue
                car_info[label] = value
                if label.lower().strip() == "marque":
                    has_marque = True
                if label.lower().strip() in ("modèle", "modéle"):
                    has_modele = True

        if has_marque and has_modele:
            return car_info
        return None
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None
    finally:
        time.sleep(DETAIL_SLEEP)

def main():
    session = make_session()
    urls = collect_item_urls(session, pages=MAX_PAGES)
    print(f"Collected {len(urls)} item URLs, fetching details with {MAX_WORKERS} workers...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_url = {ex.submit(parse_detail, u, session): u for u in urls}
        for fut in concurrent.futures.as_completed(future_to_url):
            u = future_to_url[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Worker error for {u}: {e}")

    # build columns set
    cols = set()
    for r in results:
        cols.update(r.keys())
    # exclude unwanted columns
    exclude = {"URL", "Couleur du véhicule", "Cylindrée", "Etat du véhicule"}
    cols = [c for c in cols if c not in exclude]

    # ensure stable order with Marque/Modèle/Price first
    preferred = ["Marque", "Modèle", "Price"]
    final_columns = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    # fill missing keys
    for r in results:
        for c in final_columns:
            if c not in r:
                r[c] = ""

    df = pd.DataFrame(results, columns=final_columns)
    df.to_csv("tayara_cars.csv", index=False)
    print(f"Done. Scraped {len(results)} cars. Saved to tayara_cars.csv")

if __name__ == "__main__":
    main()