# Text Sentiment Analysis
Text Sentiment Analysis in Python using Natural Language Processing (NLP) for Negative/Positive Content Predictions. Deployed on the Cloud using Streamlit on the Heroku Platform.

## Web Application Demo
[View the deployed WebApp on Heroku](https://some-app.herokuapp.com/).

## Installation : Setting up the Application Locally
* Dependencies:
  * Run the command <code>pip install -r requirements.txt</code> on your cmd/python terminal.
  * It is highly recommended to create a new [Virtual Environment](https://docs.python.org/3/library/venv.html) first before running the above commands. The instructions for doing the same is [provided below](/#creating-virtual-environments-for-python-development-in-visual-studio-code-for-this-project).
<!-- https://github.com/khanfarhan10/TextSentimentAnalysis#creating-virtual-environments-for-python-development-in-visual-studio-code-for-this-project -->
* Deployment:
  * Use Streamlit App : <code>streamlit run app.py</code>
*  If there is no popup window opening in the browser you can paste the following address : [http://localhost:8501/](http://localhost:8501).
  
# Creating Virtual Environments for Python Development in Visual Studio Code for this Project

A Guide to Creating Virtual Environments into Python and Using them Effectively.

## Clone the Text Sentiment Analysis Repo :
Head over to [our github repository](https://github.com/khanfarhan10/TextGeneration) ,fork the repo to your github account & clone the repository into your local machine.

## Initial Setup

**Open CMD/PowerShell from the VSCode Terminal :**

It should display an output like the following :

**CMD**
<code>Microsoft Windows [Version 10.0.18363.1198]
(c) 2019 Microsoft Corporation. All rights reserved.
C:\Users\farha\Documents\GitHub\TextGeneration></code>

or

**Powershell**
<code>
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.
Try the new cross-platform PowerShell https://aka.ms/pscore6</code>

**Check for the python version you're running :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>python --version
Python 3.7.6</code>

**Check for the Python Packaging Index (Pypi) version you're running :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>pip --version
pip 20.0.2 from C:\ProgramData\Anaconda3\lib\site-packages\pip (python 3.7)</code>

**Install the virtualenv module from pip :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>pip install virtualenv</code>

**Create a project environment directory with YourAwesomeProjectNameEnvironment :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>mkdir TextGenEnv</code>

**Create a new (empty) virtual environment in YourAwesomeProjectNameEnvironment :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>virtualenv TextGenEnv</code>

###### Note : If you have problems with this step, try followng the debugging options provided below.

**Enter into the newly created (empty) virtual environment in YourAwesomeProjectNameEnvironment :**
<code>C:\Users\farha\Documents\GitHub\TextGeneration>TextGenEnv\Scripts\activate</code>

You will notice a (YourAwesomeProjectNameEnvironment) appearing in the Command Line Interface :
<code>(TextGenEnv) C:\Users\farha\Documents\GitHub\TextGeneration></code>

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
<code>streamlit run TextGenApp.py</code>

### Useful Links for Debugging :

- https://github.com/ContinuumIO/anaconda-issues/issues/10822
- https://dev.to/idrisrampurawala/setting-up-python-workspace-in-visual-studio-code-vscode-149p
- https://dev.to/idrisrampurawala/flask-boilerplate-structuring-flask-app-3kcd

# Voila Magic!
