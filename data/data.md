## Web Scraping

In the [web_scrape.py](web_scrape.py) file I am currently targeting NPR.org to generate news article data. The script generates urls that I will read in my next week of work in order to pull some of the article out to make the data for my machine learning model. I have it pulling a little under 500 urls, but I can adjust the parameters slightly to get more.

The [urls.csv](urls.csv) file is the output of the above script and is what I will use to generate the data.

The current raw data folder is samples from premade datasets
https://www.kaggle.com/code/thedrcat/predict-with-daigt-v3-train-dataset
https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview