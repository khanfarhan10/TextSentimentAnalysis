<center><a href="https://winterofcode.com/"><img src="https://camo.githubusercontent.com/c73f77959233a8adb69f3dee7bbb3ba5e016f4239c7496c82538cc60c984f56e/68747470733a2f2f77696e7465726f66636f64652e636f6d2f7374617469632f6d656469612f6f72672d6c6f676f2e39333564376634382e706e67" alt="gsoc" height="50"/></a>
<a href="https://www.python.org/"><img src="https://www.python.org/static/community_logos/python-logo.png" height="45"/></a>
<a href="https://fury.gl/latest/community.html"><img src="https://raw.githubusercontent.com/divyake/Cysec-Hacktoberfest/dcc84465cfcff73981f8fcb5c8fe3b1710c007e1/assets/logo.svg" alt="DSC-IEM" height="45"/></a>
</center>

# Winter of Code Final Work Product
* **Name:** DINESH KUMAR
* **Organisation:** DSC-IEM
* **Project:** [TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis)

## Proposed Objectives
* Selection of proper dataset
* Preprocessing of data including null values and removing duplicates values.
* Featurisation and EDA
* Model selection and tuning

## Modified Objectives
* Selection of proper dataset
* Preprocessing of data including null values
* Featurisation and EDA
* Wordcloud
* Tf-Idf 
* Model selection and tuning


## Objectives Completed
* ### Selection of proper Dataset 
Getting proper data for training models suitable to our requirements is important.Therefore, I picked up the well known "Amazon fine food reviews" dataset from the kaggle.

* ### Data preprocessing
I have distributed score value in three partsnegative(score>3),Positive(score>3),neutral(score==3) and also cleaned text and removed html tags and unnecessary words.

* ### Featurisation, EDA and Tf-idf
 I have done **EDA** using piecharts and barcharts, before Featurisation firstly I split my dataset into training and test sets,It is best practice to split the data Before we do text featurization.then i applied **Tf-idf**for Featurisation.


* ### Model selection
 Of all the models tried, **Naive Bayes**works the best after vectorisation giving the accuracy of 85% and F1 score of 78% around which is pretty good as a baseline model.

* ### Deployed It on MY local server
For checking my model performance i used flask and deployed it on my local server and it is working nicely and precisely.Giving good results.

  *Pull Requests:*
  * **Model Improvised:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/18

## Objectives in Progress
* ### Deploy the model Globally using Heroku

  I plan to further save this model and deploy it on heroku so that it can be access by anybody globally. 