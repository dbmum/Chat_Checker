import read_urls, web_scrape

def main():
    """
    Running this will update the urls and the parquet file
    """
    # Scrape article urls to read
    # Change False to True in order to create the urls file from scratch
    web_scrape.main(False)
    # read the articles as a parquet file
    read_urls.main()

if __name__ == '__main__':
    main()