import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_list_url = "https://www.automobile.tn/fr/occasion/"
base_site_url = "https://www.automobile.tn"
headers = {"User-Agent": "Mozilla/5.0"}

all_car_data = []
all_columns = set()

for page in range(1, 180):
    print(f"Scraping listing page {page}")
    resp = requests.get(f"{base_list_url}{page}", headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    articles_div = soup.find("div", class_="articles")
    if not articles_div:
        continue

    car_links = articles_div.select("a.occasion-link-overlay")
    for a in car_links:
        car_url = base_site_url + a["href"]
        print(f"Scraping {car_url}")

        car_resp = requests.get(car_url, headers=headers)
        car_soup = BeautifulSoup(car_resp.text, "html.parser")

        car_info = {"URL": car_url}

        # PRICE
        price_div = car_soup.select_one("div.price-box div.price")
        price = ""
        if price_div:
            price = price_div.get_text(strip=True).replace("\n", " ")
        car_info["Price"] = float(price[:-2].replace(" ", "")  )
        all_columns.add("Price")
        print(f"Price: {price}")

        # MAIN SPECS
        main_specs_div = car_soup.find("div", class_="main-specs")
        if main_specs_div:
            for li in main_specs_div.select("li"):
                name_node = li.select_one("span.spec-name")
                value_node = li.select_one("span.spec-value")
                if name_node and value_node:
                    key = name_node.get_text(strip=True)
                    value = value_node.get_text(" ", strip=True)
                    car_info[key] = value
                    all_columns.add(key)

        # DIVIDED & CHECKED SPECS
        for box in car_soup.select("div.box"):
            title_node = box.select_one("div.box-inner-title")
            if not title_node:
                continue

            title = title_node.get_text(" ", strip=True)

            # Divided specs
            divided_div = box.select_one("div.divided-specs")
            if divided_div:
                for li in divided_div.select("li"):
                    name_node = li.select_one("span.spec-name")
                    value_node = li.select_one("span.spec-value")
                    if name_node and value_node:
                        key = f"{title} - {name_node.get_text(strip=True)}"
                        value = value_node.get_text(" ", strip=True)
                        car_info[key] = value
                        all_columns.add(key)

            # Checked specs 
            checked_div = box.select_one("div.checked-specs")
            if checked_div:
                for li in checked_div.select("li"):
                    value_node = li.select_one("span.spec-value")
                    if value_node:
                        key = f"{title} - {value_node.get_text(strip=True)}"
                        car_info[key] = "Yes"
                        all_columns.add(key)

        all_car_data.append(car_info)
        time.sleep(0.5)

all_columns = list(all_columns)
final_columns = ["URL", "Price"] + [c for c in all_columns if c not in ["URL", "Price"]]

for car in all_car_data:
    for col in final_columns:
        if col not in car:
            if " - " in col:       
                car[col] = "No"
            else:
                car[col] = ""

df = pd.DataFrame(all_car_data, columns=final_columns)
df.to_csv("used_cars_specs.csv", index=False)
print("Done, CSV saved")
