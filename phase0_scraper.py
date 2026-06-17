import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_pdfs():
    url = "https://www.dra.gov.pk/category/publications/meeting_minutes/registration_board/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    save_dir = "dra_meeting_minutes"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Fetching page: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch the webpage: {e}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    
    pdf_links = []
    for link in links:
        href = link['href']
        if href.lower().endswith('.pdf'):
            full_url = urljoin(url, href)
            if full_url not in pdf_links:
                pdf_links.append(full_url)
                
    print(f"Found {len(pdf_links)} PDF files to download.")
    
    for i, pdf_url in enumerate(pdf_links, 1):
        # Extract filename from URL or assign a default
        filename = pdf_url.split("/")[-1]
        # In case there are query parameters after .pdf
        if '?' in filename:
            filename = filename.split('?')[0]
        if not filename.lower().endswith('.pdf'):
            filename = f"document_{i}.pdf"
            
        filepath = os.path.join(save_dir, filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            print(f"File already exists (skipping): {filename}")
            continue
            
        print(f"Downloading [{i}/{len(pdf_links)}]: {filename}...")
        try:
            pdf_response = requests.get(pdf_url, headers=headers, stream=True)
            pdf_response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")

if __name__ == "__main__":
    scrape_pdfs()
