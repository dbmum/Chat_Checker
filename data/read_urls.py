import csv
import requests
from bs4 import BeautifulSoup, Comment, Tag
import pandas as pd

def scrape_article(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extracting article title
            article_title = soup.find("div", class_="storytitle").find("h1").text.strip()

            # Extracting article body
            article_body = soup.find("div", id="storytext")

            # Exclude divs with class 'credit-caption'
            for div in article_body.find_all("div", class_="credit-caption"):
                div.extract()

            # Finding all paragraphs
            paragraphs = article_body.find_all("p")
            article_text = ""
            for p in paragraphs:
                # Extracting text from paragraph excluding links and nested tags
                paragraph_text = ''.join([child.strip() if isinstance(child, str) else ' ' + child.text.strip() for child in p.contents if not isinstance(child, Comment)])
                article_text += paragraph_text.strip() + " "
            # Getting the first 200 words of the article
            article_text = ' '.join(article_text.split()[:200])
            return article_title, article_text
        else:
            print(f"Failed to fetch URL: {url}, Status code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred while scraping URL: {url}, Error: {e}")
    return None, None

def main():
    input_csv = 'data/urls.csv'  # Change this to the path of your input CSV file
    output_parquet = 'data/articles.parquet'  # Change this to the path of the output Parquet file

    articles_data = []
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            url = row[0]  # Assuming URLs are in the first column
            print(f"Scraping article from URL: {url}")
            article_title, article_text = scrape_article(url)
            if article_title and article_text:
                articles_data.append([article_title, article_text, url])

    # Convert data to DataFrame
    df = pd.DataFrame(articles_data, columns=["Title", "Content", "URL"])

    # Save DataFrame to Parquet file
    df.to_parquet(output_parquet)

    print("Scraping complete and saved as Parquet file!")

# run just one url
def _main():
    url = "https://www.npr.org/2024/01/10/1223730333/bottled-water-plastic-microplastic-nanoplastic-study"

    print(f"Scraping article from URL: {url}")
    article_title, article_text = scrape_article(url)
    if article_title and article_text:
        print(article_title)
        print(article_text)

if __name__ == "__main__":
    main()