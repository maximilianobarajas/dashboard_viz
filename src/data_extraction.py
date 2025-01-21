from threading import Lock
from mudafy_scrapper import SequentialIDGenerator,determine_total_pages,fetch_and_parse_page
from concurrent.futures import ThreadPoolExecutor, as_completed

from concurrent.futures import ThreadPoolExecutor, as_completed
from mudafy_scrapper import SequentialIDGenerator
shared_properties_lock = Lock()
id_generator = SequentialIDGenerator()
delegaciones = [
    "azcapotzalco",

]



"""
"benito-juarez",
"coyoacan",
"cuajimalpa-de-morelos",
"gustavo-a-madero",
"miguel-hidalgo",
"tlahuac",
"iztacalco",
"iztapalapa",
"venustiano-carranza",
"la-magdalena-contreras",
"alvaro-obregon",
"cuauhtemoc"
"""
shared_properties = {}

for delegacion_de_la_fuente in delegaciones:
    base_url = f"https://mudafy.com.mx/venta/propiedades/cdmx-{delegacion_de_la_fuente}"

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        num_pages = determine_total_pages(base_url,delegacion_de_la_fuente)
        for page in range(1, num_pages):
            url = f"{base_url}/{page}-p"
            futures.append(executor.submit(fetch_and_parse_page, url, id_generator, shared_properties, delegacion_de_la_fuente))
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Successfully processed a page for {delegacion_de_la_fuente}.")
            except Exception as e:
                print(f"Error during page fetch for {delegacion_de_la_fuente}: {e}")

print("Extraction complete.")
import json
import datetime

def save_to_jsonl(data, filename):
    """Saves a list of dictionaries to a JSONL file with UTF-8 encoding.

    Args:
        data: A list of dictionaries to save.
        filename: The name of the file to save to.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    properties_list = list(shared_properties.values())
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"{current_date}.jsonl"
properties_list = list(shared_properties.values())
current_date = datetime.date.today().strftime("%Y-%m-%d")
filename = f"{current_date}.jsonl"
save_to_jsonl(properties_list, filename)