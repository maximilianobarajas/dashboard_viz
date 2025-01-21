import requests
from bs4 import BeautifulSoup
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
class SequentialIDGenerator:
    def __init__(self, prefix="mdf-"):
        self.prefix = prefix
        self.counter = 1
        self.lock = threading.Lock()

    def get_next_id(self):
        with self.lock:
            unique_id = f"{self.prefix}{self.counter:010d}"
            self.counter += 1
        return unique_id
def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.lower()
import re
from bs4 import BeautifulSoup
import time
import re
def extract_property_details(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch property details for {url}")
        return {}
    soup = BeautifulSoup(response.text, "html.parser")
    details = {
        "superficie_de_terreno": 0,
        "superficie_construida": 0,
        "recamaras": 0,
        "baños": 0,
        "estacionamiento": 0,
        "google_maps_url": None
    }
    breadcrumb_section = soup.find("section", class_="breadcrumb")
    if breadcrumb_section:
        breadcrumb_items = breadcrumb_section.find_all("li")
        if breadcrumb_items:
            last_item = breadcrumb_items[-4]
            span_tag = last_item.find("span")
            if span_tag and span_tag.text.strip():
                details["delegacion"] = normalize_text(span_tag.text.strip())
    for div in soup.find_all("div"):
        img_tag = div.find("img")
        if img_tag and img_tag.get("alt"):
            key = img_tag.get("alt").strip()
            p_tag = div.find("p")
            if p_tag:
                value_match = re.search(r'\d+', p_tag.get_text(strip=True))
                if value_match:
                    value = int(value_match.group())
                else:
                    value = 0
                if key == "Superficie total":
                    details["superficie_construida"] = value
                if key == "Superficie del terreno":
                    details["superficie_de_terreno"] = value
                if key == "Dormitorios":
                    details["recamaras"] = value
                if key == "Baños":
                    details["baños"] = value
                if key == "Cocheras":
                    details["estacionamiento"] = value
    return details
def fetch_and_parse_page(url, id_generator, shared_properties, delegacion_de_la_fuente):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page: {url}")
            return
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tags = soup.find_all('script', type="application/ld+json")
        page_properties_found = False
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if "@type" in data and data["@type"] in ["SingleFamilyResidence", "Product"]:
                    offers = data.get("offers", {})
                    offer = offers[0] if isinstance(offers, list) else offers
                    url = offer.get("url", "URL not available").strip()
                    description = data.get("description", "No description available").strip()
                    price = offer.get("price", "Price not available")
                    if url != "URL not available" and description != "No description available" and price != "Price not available":
                        property_details = extract_property_details(url)
                        unique_id = id_generator.get_next_id()
                        with threading.Lock():
                            shared_properties[unique_id] = {
                                "id": unique_id,
                                "delegacion": delegacion_de_la_fuente,
                                "precio": price,
                                "superficie_de_terreno": property_details.get("superficie_de_terreno", 0),
                                "superficie_construida": property_details.get("superficie_construida", 0),
                                "recamaras": property_details.get("recamaras", 0),
                                "baños": property_details.get("baños", 0),
                                "estacionamiento": property_details.get("estacionamiento", 0),
                                "campos_adicionales": {
                                    "url": url,
                                    "descripcion": description,
                                }
                            }
                            page_properties_found = True
            except Exception as e:
                print(f"Error parsing script content: {e}")
        return page_properties_found
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
def determine_total_pages(url,delegacion_de_la_fuente):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch initial page for determining total pages: {url}")
            return 0
        soup = BeautifulSoup(response.text, "html.parser")
        pagination_links = soup.find_all('a', href=re.compile(r"/venta/propiedades/cdmx-"+delegacion_de_la_fuente +"/\d+"))
        page_numbers = []
        for link in pagination_links:
            match = re.search(r"/(\d+)", link['href'])
            if match:
                page_numbers.append(int(match.group(1)))

        if page_numbers:
            total_pages = max(page_numbers)
            print(f"Total pages determined: {total_pages}")
            return total_pages
        print("Could not determine total pages dynamically.")
        return 0
    except Exception as e:
        print(f"Error during pagination determination: {e}")
        return 0
