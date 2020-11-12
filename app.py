# streamlit run SentimentAnalysisApp.py

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# for crunching github data
import requests
import json
import time
import datetime
import os
import fnmatch
import pandas as pd

# NLP Packages
from textblob import TextBlob
import random 
import time

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

from keras.layers import LSTM

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
# a load function with
# @st.cache
# list out the loading weight functionalities for the models :
@st.cache
def load_intermediate():
    files=find("adam_acc_8643",  os.getcwd())
    # model=keras.models.load_model("/content/adam_acc_8643.h5")
     model=keras.models.load_model(files[0])
    return model

def model_TextBlob(text_input):
    blob = TextBlob(str(text_input))
    return (blob.sentiment.polarity+1)/2

def get_fixed_word_to_id_dict(): 
    INDEX_FROM=3   # word index offset
    
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id[" "] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    return word_to_id

def decode_to_sentence(data_point): 
    NUM_WORDS=1000 # only use top 1000 words
    
    word_to_id = get_fixed_word_to_id_dict()

    id_to_word = {value:key for key,value in word_to_id.items()}
    return ' '.join(id_to_word[id] for id in data_point )

def encode_sentence(sent): 
    # print(sent)
    encoded = []
    
    word_to_id = get_fixed_word_to_id_dict() 
    
    for w in sent.split(" "): 
        if w in word_to_id: 
            encoded.append(word_to_id[w])
        else: 
            encoded.append(2)        # We used '2' for <UNK> 
    return encoded 



def model_KerasIntermediate(text_input):
    np_load_old = np.load   # save old function for calling later 

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True)
    
    print('Build model...')

    l="""
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    """
    model=load_intermediate()
    
    test_sentences = [] 

    test_sentence = str(text_input)
    test_sentence = encode_sentence(test_sentence)
    test_sentences.append(test_sentence) 
    test_sentences = sequence.pad_sequences(test_sentences, maxlen=400)
    predictions = model.predict(test_sentences)
    return predictions[0]

def predictor_page():
    # this page will be responsible for dealing with the predictions

    # decorations
    html_temp = """
    <div style="background-color:#000000;padding:10px">
    <h2 style="color:white;text-align:center;"><b>Text Sentiment Analysis Predictor</b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""<br>""", unsafe_allow_html=True)

    MODELS={"Basic":model_TextBlob,"Intermediate":model_KerasIntermediate,"Complex":model_KerasIntermediate}
    # allow choice of models
    option = st.selectbox('Choose an NLP Model Complexity:', MODELS.keys)
    current_model=MODELS[option]
    
    random_text_string="Hello World"
    # warming up the model
    
    results=current_model(random_text_string)
    # run the model load functions here:
    

    # create text input
    text_label = "Text to Analyze"
    filled_text = "Enter your text here"
    ip = st.text_input(text_label, filled_text)  # , height=600 not merged yet
    show_preds = False
    # print(ip)

    show_preds = False if ((ip == filled_text) or (ip == "")) else True
    # print(show_preds)

    rm = """
    # for instant predictions or lazy predictions
    # perform predictions instantly or wait for user to press a button
    inst = st.checkbox("Instantaneous Predictions", True)
    if inst is False:
        preds=st.button("Predict")
    """

    # show results
    if (show_preds == True):
        positivity_scale = current_model(ip)
        # perform thresholding methods
        p=positivity_scale
        result=["a" if p<=0.25 else "b" if p<=0.50 else "c" if p<=0.75 else "Hate"]
        result+= " Speech"
        st.success('Predicted Text Class : {}'.format(result))
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(positivity_scale):
            latest_iteration.text(f'Measured Positivity : {i+1}')
            bar.progress(i + 1)

        st.balloons()

    # st.write("Note: High Complexity/Long Text Inputs may be computationally expensive and might lead to delayed processes & performance issues.")
    st.markdown("##### Note: High Complexity/Long Text Inputs may be computationally expensive and might lead to delayed processes & performance issues.")


def about_page():
    # the about the app page with aims & contributors
    print("Hello World!")
    # st.text("hello world")
    components.html(
        """
    <div style="background-color:#ff0055;padding:10px">
    <h1 style="color:white;text-align:center;">About the Project</h1>
    </div>
    
    """
    )
    img_top = """<center><img src="https://i.imgur.com/yOS7IGv.png" width="700px"></center>"""
    st.markdown(img_top, unsafe_allow_html=True)
    topic = """
    
    <br>
    
    The main motive of this project is to predict the amount of area that can get 
    burned in a forest fire based on some parameters like `Humidity(RH)`, `Wind(wind)`,`Rain(rain)`, 
    `Temperature(temp)` etc. 
    
    The project is a part of Hacktoberfest contribution and it has been initiated by <a href="https://github.com/dsc-iem">DSC-IEM</a> .
    We used different Model Building techniques for building the model and did an in-depth exploratory analysis 
    of the provided data. And except these things, creating a user-friendly web-app and deploying it in cloud is 
    also an integral part of a Data Science life cycle. So, we also have put together this web-app to show that.
    
    <p style="color:blue;">If you liked this project then it will be really motivating for us if you can star our repository😄.</p>
     
    
    <br>
    
    [![ReadMe Card](https://github-readme-stats.vercel.app/api/pin/?username=soumya997&repo=AI-Hacktoberfest&theme=light)](https://github.com/dsc-iem/AI-Hacktoberfest)

    """

    st.markdown(topic, unsafe_allow_html=True)



def is_update_req(current_time):
    txt_file = find(current_time,  os.path.join(ROOT_DIRECTORY, "temp_data"))
    # print (txt_file)
    return True if len(txt_file) == 0 else False

def get_simple_contribs():
    page_link = "https://api.github.com/repos/dsc-iem/AppDev-Hacktoberfest/stats/contributors"
    repo = requests.get(page_link).json()
    return repo

def update_file():
    # create save and delete old
    # find the ordered list of contributors
    # page_link = "https://api.github.com/repos/dsc-iem/AI-Hacktoberfest/stats/contributors"
    # https://github.com/dsc-iem/AppDev-Hacktoberfest
    # page_link = "https://api.github.com/repos/khanfarhan10/khanfarhan10.github.io/stats/contributors"
    page_link = "https://api.github.com/repos/dsc-iem/AppDev-Hacktoberfest/stats/contributors"
    repo = requests.get(page_link).json()
    # check if rate limit exceeded
    # print(type(repo))
    # print(len(repo))
    # print(repo[0])
    exceeded = True if "message" in repo else False
    # print(exceeded)

    if exceeded == True:
        print(
            "Rate Limit Exceeded, Update Cancelled ! Using Last Updated Contributors List.")
        return 0

    # delete the old files
    old_files = find("*.txt", os.path.join(ROOT_DIRECTORY, "temp_data"))
    for eachfile in old_files:
        os.remove(eachfile)

    writer(time=get_hourly_time(), json_dict=repo)
    return 1


def writer(time=0, json_dict=None):
    with open('temp_data/data'+str(time)+'.txt', 'w') as outfile:
        json.dump(json_dict, outfile)


def get_hourly_time(now=datetime.datetime.now()):
    timestamp = str(now.year) + str(now.month)+str(now.day)+str(now.hour)
    return timestamp


def maintainers_page():
    return 1


def collaborator_page():
    json_file=get_simple_contribs()
    repo = json.load(json_file)
        for each_contrib in repo:
            Total_Commits = each_contrib["total"]
            weeks = each_contrib["weeks"]
            additions = 0
            deletions = 0
            for each_week in weeks:
                additions += each_week["a"]
                deletions += each_week["d"]
            author_details = each_contrib["author"]
            author_name = author_details["login"]
            # print(Total_Commits, additions, deletions)
            # print(author_name)
            df = df.append({'name': author_name, 'commits': Total_Commits,
                            'adds': additions, 'dels': deletions}, ignore_index=True)
            # objective get username, number of commits, additions, & deletions

    df = df.astype(int, errors='ignore')
    df = df.sort_values(by=['commits', 'adds'], ascending=False)
    
    """
    # print(ROOT_DIRECTORY)
    # print(get_hourly_time())
    # print(len([]))
    if is_update_req(get_hourly_time()):
        update_file()
    # update_file()
    # extracting information
    files = find("*.txt",  os.path.join(ROOT_DIRECTORY, "temp_data"))
    df = pd.DataFrame()
    with open(files[0]) as json_file:
        repo = json.load(json_file)

        for each_contrib in repo:
            Total_Commits = each_contrib["total"]
            weeks = each_contrib["weeks"]
            additions = 0
            deletions = 0
            for each_week in weeks:
                additions += each_week["a"]
                deletions += each_week["d"]
            author_details = each_contrib["author"]
            author_name = author_details["login"]
            # print(Total_Commits, additions, deletions)
            # print(author_name)
            df = df.append({'name': author_name, 'commits': Total_Commits,
                            'adds': additions, 'dels': deletions}, ignore_index=True)
            # objective get username, number of commits, additions, & deletions

    df = df.astype(int, errors='ignore')
    df = df.sort_values(by=['commits', 'adds'], ascending=False)
    # print(df.head(10))
    """
    # displaying info
    # divide to even or even + 1 for

    headings = """
        <div style="background-color:#000000;padding:10px">
        <h1 style="color:white;text-align:center;">Project Collaborators:</h1>
        </div>
        """

    components.html(headings)

    odd = True if len(df) % 2 == 1 else False

    first = """
        <html>
            <head>
                
            </head>

            <body>
                <a>"""
    mid = """
    <strong style="font-size:20px">
                        <pre class="tab">{0} <a style="font-size:14px">{1} commits </a><a style="color: #2bff00;font-size:10px">{2}++ </a><a style="color: #FF0000;font-size:10px">{3}--</a>{8}{4} <a style="font-size:14px">{5} commits </a><a style="color: #2bff00;font-size:10px">{6}++ </a><a style="color: #FF0000;font-size:10px">{7}--</a></pre>
                        <div class="github-card" data-github="{0}" data-width="350" data-height="150" data-theme="default"></div>
                        <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                        <div class="github-card" data-github="{4}" data-width="350" data-height="" data-theme="default"></div>
                        <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                    </strong> 
    """
    end = """                
            </body>
        </html>


        


        """

    text = """"""

    text += first

    for i in range(0, len(df), 2):
        # print("Row :",i,df.iloc[i,:])
        first_row = df.iloc[i, :]
        second_row = df.iloc[i+1, :]
        # print(type(first_row))
        # num_spaces = 30-len(first_row["name"])-len(second_row["name"])
        num_spaces = 47-len(first_row["name"])-len(second_row["name"]) - len(str(first_row["commits"]))-len(str(second_row["commits"]))
        num_spaces = num_spaces-len(str(first_row["adds"]))-len(str(second_row["adds"]))-len(str(first_row["dels"]))-len(str(second_row["dels"]))
        num_spaces = 9 if num_spaces < 0 else num_spaces  # default spacing
        spaces = " "*num_spaces
        middle = mid.format(first_row["name"], first_row["commits"], first_row["adds"], first_row["dels"],
                            second_row["name"], second_row["commits"], second_row["adds"], second_row["dels"], spaces)
        text += middle

    text += end

    # odd = True
    # Handle odd cases
    if odd:
        alone = """
        <strong style="font-size:20px">
                            <pre class="tab">{0} <a style="font-size:14px">{1} commits </a><a style="color: #2bff00;font-size:10px">{2}++ </a><a style="color: #FF0000;font-size:10px">{3}--</a></pre>
                            <div class="github-card" data-github="{0}" data-width="350" data-height="150" data-theme="default"></div>
                            <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                        </strong> 
        """
        last_row = df.iloc[len(df)-1, :]
        text += alone.format(last_row["name"], last_row["commits"],
                             last_row["adds"], last_row["dels"])
        # print("I love coding")

    # print(text)
    components.html(text, height=700, scrolling=True, width=800)


# Sidebar Editing with utility fucntions in argument : # TODO
def sidebar_page(pages, header_img=None, title=None, text=None):
    if header_img != None:
        st.sidebar.markdown(header_img, unsafe_allow_html=True)





def sidebar_nav():
    # top image for DSC-IEM
    html_img = """<center><img src="https://i.imgur.com/mx9aCum.png" width="300px" ></center>"""
    st.sidebar.markdown(html_img, unsafe_allow_html=True)
    # https://i.ibb.co/VY5wCkN/47480912-png.png
    # height="130px"

    # NAV BAR
    st.sidebar.markdown("""## Navigation Bar: <br> """, unsafe_allow_html=True)
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    current_page = st.sidebar.radio(
        " ", ["Predictions",  "Project Collaborators", "About"])


    # st.markdown("""<br></br> <br>""",unsafe_allow_html=True)
    sidetext = """
    <br><br><br><br><br>Thank you for visiting this website🤗.  
    We contribute towards open source :  
    Feel free to visit [our github repository](https://github.com/khanfarhan10/TextSentimentAnalysis)
    """

    st.sidebar.markdown(sidetext, unsafe_allow_html=True)


    all_pages = {"Predictions": predictor_page,
                "Project Collaborators": collaborator_page, "About": about_page}
    # predictor_page()

    func = all_pages[current_page]
    func()

    # http://ffp-web-app.herokuapp.com/
    
ROOT_DIRECTORY = os.getcwd()
sidebar_nav()