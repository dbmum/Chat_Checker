import requests
from bs4 import BeautifulSoup
import csv

BASE_URL = "https://www.npr.org"

def main():

    url_extensions = ['/sections/National', '/sections/World', '/sections/Politics', '/sections/Business', '/sections/Climate', '/sections/Science', '/sections/Health', '/sections/Race']
    urls = set()

    for url_extension in url_extensions:
        print(url_extension)
        url = BASE_URL + url_extension
        scrape_articles_from_url(urls, url, True)
    
    with open('data/urls.csv','w',newline="") as f:
        writer = csv.writer(f)
        for url in urls:
            writer.writerow([url])

    print(f'Found {len(urls)} urls!')


def scrape_articles_from_url(urls: set, url: str, primary: bool):
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
                        scrape_articles_from_url(urls, BASE_URL + href, False)
                
    else:
        print(f"\nFailed to retrieve {url}. \nStatus code: {response.status_code}\n")


if __name__ == "__main__":
    main()