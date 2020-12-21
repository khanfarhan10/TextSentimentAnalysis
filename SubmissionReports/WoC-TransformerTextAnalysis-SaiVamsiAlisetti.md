<center><a href="https://winterofcode.com/"><img src="https://camo.githubusercontent.com/c73f77959233a8adb69f3dee7bbb3ba5e016f4239c7496c82538cc60c984f56e/68747470733a2f2f77696e7465726f66636f64652e636f6d2f7374617469632f6d656469612f6f72672d6c6f676f2e39333564376634382e706e67" alt="gsoc" height="50"/></a>
<a href="https://www.python.org/"><img src="https://www.python.org/static/community_logos/python-logo.png" height="45"/></a>
<a href="https://fury.gl/latest/community.html"><img src="https://raw.githubusercontent.com/divyake/Cysec-Hacktoberfest/dcc84465cfcff73981f8fcb5c8fe3b1710c007e1/assets/logo.svg" alt="DSC-IEM" height="45"/></a>
</center>

# Winter of Code Final Work Product
* **Name:** Alisetti Sai Vamsi
* **Organisation:** DSC-IEM
* **Project:** [TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis)

## Proposed Objectives
* Transformer Model Construction
* Voice Recognition
* CLI Tool
* KeyWord Extraction

## Modified Objectives
* Voice Recognition on frontend in js
* CLI Tool with the Transformer Model
* KeyWord Extraction with spacy
* DistilBert Trained on Amazon Food Review Dataset
* Connecting flask to streamlit for lighter api inference

## Objectives Completed
* ### Voice Recognition on frontend in js

  Made use of the WebAPI of the chrome v8 browser engine to create a speech recognition script that transforms audio signal to text.

  *Pull Requests:*
  * **Speech to text #17:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/17

* ### Connecting flask to streamlit for lighter api inference

  Integrated streamlit with flask using the requests library of python transforming streamlit as a frontend framework and flask web api as a backend microservice.

  *Pull Requests:*
  * **Streamlit to flask connection #16:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/16

* ### Research on dataset and model selection
    
  For the dataset my primary selections were Standford Sentiment Treebank or the IMDB Dataset. But upon the suggestion from my peers and further research, I have chosen the amazon food reviews dataset since it has a good amount of variance and its a huge dataset.
  Since the training model was told to be lightweight I chose the distilbert transformer model leveraging the huggingface transformer library.  

* ### Training transformer model
  
  Training was done in pytorch, and the relevant hyperparameters are included in the training script. For more details please look into the training script.
  
  Training Script: https://colab.research.google.com/drive/1ejVSWQng9chJRoqWprfxTnZnsKAcRFT-?usp=sharing


## Objectives in Progress

* ### Key Word Extraction

  Currently, trying to understand the spacy library to use it efficiently for keyword extraction.


## Other Objectives
* ### Deployment to Android

  Flutter UI to me made matching the aesthetics of the web UI and also having the same functionality.

* ### Models for paraphrasing and summarization

  Idea is to leverage T5 transformer model for both these tasks as it can perform all the NLP tasks with the right way of training as presented in the T5 paper.

