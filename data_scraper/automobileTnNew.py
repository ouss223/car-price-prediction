import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#all are supposeed to be 2025 models

base_list_url = "https://www.automobile.tn/fr/neuf/recherche/s=?page="
base_site_url = "https://www.automobile.tn"

headers = {"User-Agent": "Mozilla/5.0"}

all_car_data = []
all_columns = set()

for page in range(1, 15): 
    print(f"Scraping page {page}")
    resp = requests.get(f"{base_list_url}{page}", headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    car_links = soup.select("div.versions-item a")
    for a in car_links:
        car_url = base_site_url + a["href"]
        print(f"Scraping {car_url}")

        parts = a["href"].strip("/").split("/")
        brand = parts[2] if len(parts) > 2 else ""
        model = parts[3] if len(parts) > 3 else ""

        car_resp = requests.get(car_url, headers=headers)
        car_soup = BeautifulSoup(car_resp.text, "html.parser")

        versions_table = car_soup.find("table", class_="versions")
        version_links = []

        if versions_table:
            for tr in versions_table.find_all("tr"):
                td_version = tr.find("td", class_="version")
                if td_version:
                    a_tag = td_version.find("a")
                    if a_tag:
                        version_links.append((a_tag["href"], td_version.get_text(strip=True)))
        else:
            version_links.append((a["href"], ""))

        for vlink, version_text in version_links:
            full_url = base_site_url + vlink
            vparts = vlink.strip("/").split("/")
            option = vparts[4] if len(vparts) > 4 else ""

            v_resp = requests.get(full_url, headers=headers)
            v_soup = BeautifulSoup(v_resp.text, "html.parser")

            specs_div = v_soup.find("div", id="specs")
            
            price = ""
            for div in v_soup.find_all("div"):
                div_text = div.get_text(" ", strip=True)
                if div_text.startswith("A partir de"):
                    span = div.find("span")
                    if span:
                        price = span.get_text(" ", strip=True)
                        break
            print(f"Price found: {price[0:-6]}")  

            car_info = {
                "URL": full_url,
                "Brand": brand,
                "Model": model,
                "Option": option,
                "Price": price[0:-6]  
            }

            if specs_div:
                tables = specs_div.find_all("table")
                for table in tables:
                    for row in table.find_all("tr"):
                        cols = row.find_all(["th", "td"])
                        if len(cols) >= 2:
                            key = cols[0].get_text(" ", strip=True)
                            value = cols[1].get_text(" ", strip=True)
                            if key:
                                car_info[key] = value
                                all_columns.add(key)

            all_car_data.append(car_info)
            time.sleep(0.5)  

all_columns = list(all_columns)
final_columns = ["URL", "Brand", "Model", "Option", "Price"] + [c for c in all_columns if c not in ["URL", "Brand", "Model", "Option", "Price"]]

for car in all_car_data:
    for col in final_columns:
        if col not in car:
            car[col] = ""

df = pd.DataFrame(all_car_data, columns=final_columns)
df.to_csv("automobileTnNew.csv", index=False)
print("Done, CSV saved")
