import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time

def download_images(food_item, num_images, save_folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize variables
    downloaded_images = 0
    page_number = 0

    while downloaded_images < num_images:
        # Construct the Google Images search URL for the current page
        search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={urllib.parse.quote(food_item)}&start={page_number * 20}"
        
        # Send a request to the search URL
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find image URLs
        img_tags = soup.find_all('img')
        img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]
        
        # Download images from the current page
        for img_url in img_urls:
            if downloaded_images >= num_images:
                break
            try:
                img_data = requests.get(img_url).content
                img_name = os.path.join(save_folder, f"{food_item}_{downloaded_images + 1}.jpg")
                with open(img_name, 'wb') as img_file:
                    img_file.write(img_data)
                print(f"Downloaded {img_name}")
                downloaded_images += 1
            except Exception as e:
                print(f"Could not download {img_url}: {e}")

        page_number += 1
        time.sleep(2)  

download_images("nankhatai", 50, "Indian_Food_Images/nankhatai")