import read_urls, web_scrape, create_gpt_generated_data

def main():
    """
    Running this will update the urls and the parquet file
    """
    # Scrape article urls to read
    web_scrape.main(create_file=True)
    # read the articles as a parquet file
    read_urls.main()
    # Generate ChatGPT articles **You will need your own API Key**
    create_gpt_generated_data.main()
if __name__ == '__main__':
    main()