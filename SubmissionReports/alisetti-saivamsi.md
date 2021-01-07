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

* ### Key Word Extraction

  Dependency parsing and pos tagging using the spacy library has been implemented and integrated into streamlit.

  *Pull Requests:*
  * **Inference api #27:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/27


* ### Models for paraphrasing and summarization

  Leveraged T5 for performing paraphrasing and used extractive summarization technique for summarization task. The summarization task, selectes sentences in a paragraph and calculates sentence scores based on the token score of each word in the sentence and picks the top 20 percent of the sentence with high score.

  *Pull Requests:*
  * **Model Utility #32:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/32

* ### Built a CLI Tool for processing csv files

  Made use of the same model inferences, and created a CLI application in python which can process huge csv files, and output csv files with annotated labels. This can be useful for dataset generation and processing huge chunks of data.

  *Pull Requests:*
  * **Cli #30:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/30
  * **Small changes to the CLI #31:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/31
  

## Objectives in Progress

* ### Deployment to Android
  Flutter UI to me made matching the aesthetics of the web UI and also having the same functionality.


# Developer - Winter of Code 2020
# Alisetti Sai Vamsi
### DSC - IEM : Text Sentiment Analysis

 Overview
# Contributions

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

* ### Key Word Extraction

  Dependency parsing and pos tagging using the spacy library has been implemented and integrated into streamlit.

  *Pull Requests:*
  * **Inference api #27:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/27


* ### Models for paraphrasing and summarization

  Leveraged T5 for performing paraphrasing and used extractive summarization technique for summarization task. The summarization task, selectes sentences in a paragraph and calculates sentence scores based on the token score of each word in the sentence and picks the top 20 percent of the sentence with high score.

  *Pull Requests:*
  * **Model Utility #32:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/32

* ### Built a CLI Tool for processing csv files

  Made use of the same model inferences, and created a CLI application in python which can process huge csv files, and output csv files with annotated labels. This can be useful for dataset generation and processing huge chunks of data.

  *Pull Requests:*
  * **Cli #30:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/30
  * **Small changes to the CLI #31:** https://github.com/khanfarhan10/TextSentimentAnalysis/pull/31
  
* ### Flutter Application
  Minimalistic Flutter UI made and has been integrated with flask backend.


  
# New Features

Some of the new features were:

1. CLI Tool

2. Flutter Application

3. Voice Recognition

4. Transformer models for inference


# Future Scope

There are so many avenues that this project can take up. These are some of the following which I consider to be plausible and nice:

1. Deploying the flask server onto google cloud or aws for better resources.
   
2. Publishing the flutter application and enhancing its UI.
   
3. Improving the CLI Tool further and packaging it to PIP.
   
4. Improving the current web application by using a robust frontend framework like react or angular.

# Overall Experience
The overall program was very intriguing and gave enough time for practical implementation. Also this helped learn new things and has kept me out of my comfort zone. This also grew my network, and I made some really nice friends. I want to thank the mentors especially for putting up with us. A huge shout out to them and a special shout out to Farhan for putting up with me.
