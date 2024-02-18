## Web Scraping

In the [web_scrape.py](web_scrape.py) file I am currently targeting NPR.org to generate news article data. The script generates urls that I will read in my next week of work in order to pull some of the article out to make the data for my machine learning model. I have it pulling around 1500 urls from the current news and archives. You can run the file again on different days in order to get some recent files added to the list.

The [urls.csv](urls.csv) file is the output of the above script and is what I will use to generate the data.

The [read_urls.py](read_urls.py) file reads all of the articles found at the urls in the urls file. It will make the articles.parquet file which has 'article_title' and 'article_text' columns in it.