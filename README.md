# Text Sentiment Analysis
![Python Version](https://img.shields.io/badge/Python-3.7.6-red)
![Tensorflow Version](https://img.shields.io/badge/tensorflow-2.3.0-lime)
![Keras Version](https://img.shields.io/badge/keras-2.4.3-orange)
![Pypi Version](https://img.shields.io/badge/pypi-20.0.2-yellow)
![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fkhanfarhan10%2FTextSentimentAnalysis)

![GitHub Issues Open](https://img.shields.io/github/issues-raw/khanfarhan10/TextSentimentAnalysis)
![GitHub Issues Closed](https://img.shields.io/github/issues-closed-raw/khanfarhan10/TextSentimentAnalysis)
![GitHub Open Pull Requests](https://img.shields.io/github/issues-pr-raw/khanfarhan10/TextSentimentAnalysis)
![GitHub Closed Pull Requests](https://img.shields.io/github/issues-pr-closed-raw/khanfarhan10/TextSentimentAnalysis)
![GitHub Forks](https://img.shields.io/github/forks/khanfarhan10/TextSentimentAnalysis)
![GitHub Stars](https://img.shields.io/github/stars/khanfarhan10/TextSentimentAnalysis)
![GitHub License](https://img.shields.io/github/license/khanfarhan10/TextSentimentAnalysis)

<!--
![]()
Shoutout to https://shields.io/ for these wonderful badges.
-->
Text Sentiment Analysis in Python using Natural Language Processing (NLP) for Negative/Positive Content Predictions. Deployed on the Cloud using Streamlit on the Heroku Platform.

**Winter of Code 2020 :** [View Project Ideas](https://github.com/dsc-iem/WoC-Project-Ideas#text-sentiment-analysis) or [View Issues to Solve](https://github.com/khanfarhan10/TextSentimentAnalysis/issues).

## Web Application Demo
[View the deployed WebApp on Heroku](https://some-app.herokuapp.com/).

## Installation : Setting up the Application Locally
* Dependencies:
  * Run the command <code>pip install -r requirements.txt</code> on your cmd/python terminal.
  * It is highly recommended to create a new [Virtual Environment](https://docs.python.org/3/library/venv.html) first before running the above commands. The instructions for doing the same is [provided below](#creating-virtual-environments-for-python-development-in-visual-studio-code-for-this-project).
* Deployment:
  * Use Streamlit App : <code>streamlit run TextSentimentApp.py</code>
*  If there is no popup window opening in the browser you can paste the following address : [http://localhost:8501/](http://localhost:8501).
  
# Creating Virtual Environments for Python Development in Visual Studio Code for this Project

A Guide to Creating Virtual Environments into Python and Using them Effectively.

## Clone the Text Sentiment Analysis Repo :
Head over to [our github repository](https://github.com/khanfarhan10/TextSentimentAnalysis) ,fork the repo to your github account & clone the repository into your local machine.

## Initial Setup

**Open CMD/PowerShell from the VSCode Terminal :**

It should display an output like the following :

**CMD**

<code>Microsoft Windows [Version 10.0.18363.1198]
(c) 2019 Microsoft Corporation. All rights reserved.
C:\Users\farha\Documents\GitHub\TextSentimentAnalysis></code>

or

**Powershell**

<code>
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.
Try the new cross-platform PowerShell https://aka.ms/pscore6</code>

**Check for the python version you're running :**

<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>python --version
Python 3.7.6</code>

**Check for the Python Packaging Index (Pypi) version you're running :**

<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>pip --version
pip 20.0.2 from C:\ProgramData\Anaconda3\lib\site-packages\pip (python 3.7)</code>

**Install the virtualenv module from pip :**

<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>pip install virtualenv</code>

**Create a project environment directory with YourAwesomeProjectNameEnvironment :**

<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>mkdir TextSentimentEnv</code>

**Create a new (empty) virtual environment in YourAwesomeProjectNameEnvironment :**

<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>virtualenv TextSentimentEnv</code>

Users for other python versions  may [Install Python 3.7.6](https://www.python.org/downloads/release/python-376/) first and then run the following command to choose the correct python interpreter with the correct path : 

<code>virtualenv --python=python3.7.6 TextSentimentEnv</code>

###### Note : If you have problems with this step, try followng the debugging options [provided below](#useful-links-for-debugging).

**Enter into the newly created (empty) virtual environment in YourAwesomeProjectNameEnvironment :**
<code>C:\Users\farha\Documents\GitHub\TextSentimentAnalysis>TextSentimentEnv\Scripts\activate</code>

You will notice a (YourAwesomeProjectNameEnvironment) appearing in the Command Line Interface :
<code>(TextSentimentEnv) C:\Users\farha\Documents\GitHub\TextSentimentAnalysis></code>

Wohoooo! You're now in your virtual environment.

### Install Dependencies :
Okay Great! We've got our virtualenv setup, but it's empty right now. Lets install some modules into it.

For this we will be needing a .txt file noting all the dependency requirements for a project under the project directory.

This file contains packages in the following naming fashion and can be obtained using 

<code>pip freeze > requirements.txt</code>

or using 

<code>conda list --explicit > reqs.txt</code>

When you've obtained the requirements file, do the following with your Environment Activated :
<code>pip install -r requirements.txt</code>

You are now happy to go forth coding and running your app with :
<code>streamlit run TextSentimentApp.py</code>

### Useful Links for Debugging :

- https://github.com/ContinuumIO/anaconda-issues/issues/10822
- https://dev.to/idrisrampurawala/setting-up-python-workspace-in-visual-studio-code-vscode-149p
- https://dev.to/idrisrampurawala/flask-boilerplate-structuring-flask-app-3kcd

# Voila Magic!

If you have further issues/queries regarding the project, feel free to contact us : 
- Farhan Hai Khan : njrfarhandasilva10@gmail.com
- Tannistha Pal : paltannistha@gmail.com


# Using the Streamlit App
To start using the streamlit application follow these steps:


1. Clone the repo:

```
git clone https://github.com/khanfarhan10/TextSentimentAnalysis.git
```

2. Start the streamlit app:
```
cd TextSentimentAnalysis/streamlit
streamlit run sentiment.py
```

3. Start the flask server:
```
cd TextSentimentAnalysis/server
python server.py
```

In the streamlit application, you can enter the text, and in the sidebar you can choose the task you want to perform, and hit the button for the corresponding inference.


# CLI Tool Usage

Please follow the steps to use the CLI Tool for text sentiment analysis and for summarization.

1. Clone the repo:

```
git clone https://github.com/khanfarhan10/TextSentimentAnalysis.git
```

2. Change directory into the application folder:
```
cd TextSentimentAnalysis/TxT-CLI
```

3. Install the CLI through pip:
```
pip3 install .
```

4. Usage of the CLI:

```
TxT <option> <inputfilepath> <outputfilepath>
```

5. Options
```
<option> = "tsa" (Text Sentiment Analysis)
         = "sum" (Summarization)
         
<inputfilepath> = Path to a csv file containing texts in the first column

<outputfilepath> = Path to an empty csv file
```
