# Stonks_3.0
This is a project started by the Data Science and Analytics Lab. The project focuses on Prediction of Stock Prices using Twitter Sentiment Analysis. We make API calls to the twitter API to scrap query based tweets for specified dates followed by **flair** (DistilBERT) based Sentiment Analysis of the tweet text. Further we scrap opening and closing price of stock prices for that specified timeframe. Finally we correlate the magnitude of sentiment with the rise and fall in stock prices. 
### Required Python Libraries 
>  * requests
>  * time
>  * datetime 
>  * tweepy
>  * csv
>  * pandas
>  * random
>  * numpy
>  * flair
>  * yfinance
