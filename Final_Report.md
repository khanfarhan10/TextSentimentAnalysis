<center><a href="https://winterofcode.com/"><img src="https://camo.githubusercontent.com/c73f77959233a8adb69f3dee7bbb3ba5e016f4239c7496c82538cc60c984f56e/68747470733a2f2f77696e7465726f66636f64652e636f6d2f7374617469632f6d656469612f6f72672d6c6f676f2e39333564376634382e706e67" alt="gsoc" height="50"/></a>
<a href="https://www.python.org/"><img src="https://www.python.org/static/community_logos/python-logo.png" height="45"/></a>
<a href="https://fury.gl/latest/community.html"><img src="https://raw.githubusercontent.com/divyake/Cysec-Hacktoberfest/dcc84465cfcff73981f8fcb5c8fe3b1710c007e1/assets/logo.svg" alt="DSC-IEM" height="45"/></a>
</center>

# Winter of Code Final Work Product
* **Name:** Dinesh Kumar
* **Organisation:** DSC-IEM
* **Project:** [TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis)

## Selection a Proper Dataset
* Getting proper data for training models suitable to our requirements is important.
* I have searched a lot of dataset like twitter analysis data and many more but at last i finalised **Amazon Fine Food Review** .
* I have choosed this dataset because it includes rating from **0-5 scores** for every individual review.
* The data span a period of more than 10 years, including all **~500,000 reviews** up to October 2012.
* It contains **huge** dataset due to this i have choosen this dataset for my project.
* images
![](/images/dataset.JPG)

## Data Preprocessing on dataset
* Before we move to train our model we have to do preprocessing so that we can **remove unwanted data**.
* So here in my dataset there are various columns of different values but for my project i have selected only **scores , id and reviews text** column for my project.
* after that there is score given 0-5 for reviews so i divided that into 3 categories **negative(score>3),Positive(score>3),neutral(score==3)**.
* after that i have checked that if any **duplicate values** are there than i have seen that there are duplicates values , I removed all duplicated value.
* Then i removed **html tags**, **special character** and **Tokenize** the reviews into word tokens.
* After that i split the review into words and then check if these words are in the **stop words** if so we will remove them, if not we will join.

## Featurisation , Tf-idf
* Now we have splitted my cleaned dataset into **train and test set** to work on that and build a gentle model.
* Then i featurised my dataset on tf-idf vectorizer and fit it as **tfidf_model.fit(reviews_train,sentiment_train)**.
* Then i transform it on train reviews as **reviews_train_tfidf=tfidf_model.transform(reviews_train)**.
* Then i used **WordCloud** to see **top 10 words** by importing Wordcloud in my model.
* images 
![](/images/top10_words.jpg)

## Model Selection
* Model selection is very **key point** to make your project best in term of accuracy and precission.
* after applying **EDA** on dataset i tried **three algorithms** to train my model for better prediction.
* These are 3 algorithms :- **Logistic Regression, Naive Bayes and Decision tree**.
* From these three i have figureout one algorithm which will fit to my model.
* So on the basis of **parameters** and **accuracy** i choosed Naive Bayes for my model to train my model on it.
* images
![](/images/f1_scores.jpg)

## Model Deployement
* For pre-check my model **frontend and backend** i deployed it on my local server, and it working efficiently and precisely.
* I have added feature of **prediction of sentiment** , **keywords extraction and showing** , **Polarity and Subjectivity** and **Summary**. these are some features which will comes up when we put sone text in my frontend and gives output after processing in backend.
* images of frontend
![](/images/frontend1.JPG)
![](/images/frontend2.JPG)