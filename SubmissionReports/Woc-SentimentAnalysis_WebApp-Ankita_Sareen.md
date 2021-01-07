<center><a href="https://winterofcode.com/"><img src="https://camo.githubusercontent.com/c73f77959233a8adb69f3dee7bbb3ba5e016f4239c7496c82538cc60c984f56e/68747470733a2f2f77696e7465726f66636f64652e636f6d2f7374617469632f6d656469612f6f72672d6c6f676f2e39333564376634382e706e67" alt="gsoc" height="50"/></a>
<a href="https://www.python.org/"><img src="https://www.python.org/static/community_logos/python-logo.png" height="45"/></a>
<a href="https://fury.gl/latest/community.html"><img src="https://raw.githubusercontent.com/divyake/Cysec-Hacktoberfest/dcc84465cfcff73981f8fcb5c8fe3b1710c007e1/assets/logo.svg" alt="DSC-IEM" height="45"/></a>
</center>

# Winter of Code Final Work Product
* **Name:** ANKITA SAREEN
* **Organisation:** DSC-IEM
* **Project:** [TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis)

## Proposed Objectives
* Selection of proper dataset
* Preprocessing of data including null values
* Featurisation and EDA
* Model selection and tuning

## Modified Objectives
* Selection of proper dataset
* Preprocessing of data including null values
* Featurisation and EDA
* Implementing Model selection and n-gram modelling
* Zipfile upload
* deploying using flask
* text summarisation

## Objectives Completed
* ### Selection of proper Dataset 
Getting proper data for training models suitable to our requirements is important.Therefore, I picked up the well known "Amazon fine food reviews" dataset from the kaggle.

* ### Data preprocessing
I dropped the rows where score = 3 because neutral reviews don't provide value to the prediction.Next I created a column called positivity where any score above 3 is encoded as 1 otherwise 0. For other applications I could have applied various techniques to reduce memory usage, here we are going to just drop columns which we don't require and consume deep memory.

* ### Featurisation, EDA and Tf-idf
Using the seaborn library and wordcloud,tried to analyse the prediction column and the major words for the respective sentiment.
   * Tokenization
         In order to perform machine learning on text documents, I first need to turn these text content into numerical feature vectors that Scikit-Learn can use. 
         The simplest way to do so is to use *bags-of-words*. First I converted the text document into a matrix of tokens. The default configuration tokenizes the string, by extracting words of at least 2 letters or numbers, separated by word boundaries, converts everything to lowercase and builds a vocabulary using these tokens
   * Sparse Matrix
        I then transformed the document into a bag-of-words representation i.e matrix form. The result is stored in a sparse matrix i.e it has very few non zero elements.Rows represent the words in the document while columns represent the words in our training vocabulary.
    * Used TF IDF over bag of words.
        

* ### Model selection
 Of all the models tried, Logistic Regression works the best after vectorisation giving the accuracy of 93% and ROC_AUC score of 90% around which is pretty good model.

* ### n-gram modelling

Our classifier might misclassify things like 'not good', therefore I used groups of words instead of single words. This method is called n grams (bigrams for 2 words and so on). Here I take 1 and 2 words into consideration.Basically, an N-gram model predicts the occurrence of a word based on the occurrence of its N – 1 previous words. After modelling the classifier, I could achieve the accuracy of 96% and ROC_AUC score of 93% which itself shows how good the model worked.
  Generally, the bigram model works well and it may not be necessary to use trigram models or higher N-gram models.But I tried for the trigram model in which accuracy of 97% and ROC_AUC score of 94% - a very slight increase.
     
  Training Script: https://colab.research.google.com/drive/14ekbzUx0dcEkzMdwBAhbdf4fflz6lL4H?usp=sharing
  
* ### Deploy the model in flask 
Model was further integrated with flask and then deployed on heroku with further implementation of features such as summarisation.

*Pull Requests:*
  * **Model Improvised:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/18

* ### Built a Utility Tool for  Zip File Upload
Added the feature of zip file upload, from which then csv files are extracted and predictions are made from the model. Also added one more feature, a link from where you can download the csv of the prediction dataframe generated.

  *Pull Requests:*
  * **Zip file upload #28:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/28
  * **Small changes to app.py #34:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/34
 
## Future scope
While working in the project of TextSentiment Analysis not only we got aware about more applications of NLP but we also got ideas about how to make the project more meaningful for the community around. With the new inculcated features during this period, I feel this app will certainly be of some purpose.

## Acknowledgements
I had an enriching and exciting winter of 2020, working on this project under the Winter of Code program. I was glad to be mentored by Farhan (thanks for going through all my PRs!) and Tanishtha (who helped us with her vast knowledge in and around NLP). I express my sincere thanks to all my peers working as part of WoC.

Woc was a warm and unforgettable experience for me. Everytime I talk of Open Source, TextSentiment Analysis will be remembered fondly.
