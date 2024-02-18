import requests
from bs4 import BeautifulSoup
import csv
import datetime

BASE_URL = "https://www.npr.org"
URLS_CSV_FILE = 'data/urls.csv'

def main(create_file=True):
    """
    If create_file is true it will create a new file from scratch,
    otherwise, it will just update the current file with current articles 
    and will NOT read the archive files that will not change each time
    """
    # Load existing URLs from CSV file
    existing_urls = set()
    try:
        with open(URLS_CSV_FILE, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                existing_urls.add(row[0])
    except FileNotFoundError:
        pass  # If the file doesn't exist, no need to worry

    url_extensions = ['/sections/National', '/sections/World', '/sections/Politics', '/sections/Business', '/sections/Climate', '/sections/Science', '/sections/Health', '/sections/Race']
    urls = set()

    for url_extension in url_extensions:
        print(url_extension)
        url = BASE_URL + url_extension
        scrape_articles_from_url(urls, existing_urls, url, True)
    
    if create_file:
        print('Archive files')
        # also scrape all of the articles from certain archives
        archive_urls = generate_archive_urls(2015,2023)
        for url in archive_urls:
            scrape_articles_from_url(urls,existing_urls, url, False)
        
    
    # Update CSV file with new URLs
    with open(URLS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for url in urls:
            if url not in existing_urls:  # Add only new URLs
                writer.writerow([url])

    print(f'Found {len(urls)} new URLs and updated the CSV file!')

def generate_archive_urls(start_year, end_year):
    base_url = "https://www.npr.org/sections/news/archive?date={}"
    urls = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Construct the date string in the format MM-DD-YYYY
            date_str = datetime.date(year, month, 1).strftime("%m-%d-%Y")
            # Construct the full URL
            url = base_url.format(date_str)
            urls.append(url)

    return urls

def scrape_articles_from_url(urls: set, existing_urls: set, url: str, primary: bool):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        article_blocks = soup.find_all("article")
        
        for article in article_blocks:
            links = article.find_all("a")
            
            for link in links:
                url = link.get("href")
                # make sure the article is from this century
                if url.startswith("https://www.npr.org/20"):
                    urls.add(url)
        

        # Go one level deeper away from the primary to collect more urls
        if primary:
            sublinks_to_avoid = ['Newsletter', 'Podcast', 'Contact Us', 'About Us', 'FAQ', 'School Colors']
            header_block = soup.find('header', class_="contentheader contentheader--one")
            if header_block:
                links = header_block.find_all('a')
                for link in links:
                    href = link.get("href")
                    if href.startswith('/sections') and link.text not in sublinks_to_avoid:
                        scrape_articles_from_url(urls, existing_urls, BASE_URL + href, False)
                
    else:
        print(f"\nFailed to retrieve {url}. \nStatus code: {response.status_code}\n")

if __name__ == "__main__":
    main()