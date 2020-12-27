from typing import Optional
import streamlit as st
import requests
import json
import spacy_streamlit
from spacy_streamlit.util import get_svg
from spacy import displacy


spacy_model = "en_core_web_sm"

def main():
    # Applying styles to the buttons
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

    # Heading
    st.header("Text Sentiment Analysis")
    st.sidebar.title("TxT")
    task = st.sidebar.selectbox("Choose Task: ", ("Sentiment Analysis", "Summarization", "Paraphrase"))


    # Text area for user input
    user_input = st.text_area("Enter your text here", "", height=200)



    if(task == "Sentiment Analysis"):

        if(st.button("Analyze")):
            with st.spinner('Analyze Text'):
                doc = spacy_streamlit.process_text(spacy_model, user_input)

                visual_pos(doc)
                output = forward_sentimentAnalysis(user_input)

                st.header("Sentiment")
                st.write(output)

        pass
    elif(task == "Summarization"):

        if(st.button("Summarize")):
            with st.spinner('Summarizing Text'):
                doc = spacy_streamlit.process_text(spacy_model, user_input)
                output = forward_summarization(user_input)


    elif(task == "Paraphrase"):

        if(st.button("Paraphrase")):
            with st.spinner('Paraphrasing Text'):
                doc = spacy_streamlit.process_text(spacy_model, user_input)
                output = forward_paraphrase(user_input)









def forward_sentimentAnalysis(sentence):
    # Making the request to the backend
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward", headers=headers,
                      data=json.dumps({'sentence': sentence}))
    data = r.json()
    return data["data"]


def forward_summarization(sentence):
    # Making the request to the backend
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward", headers=headers,
                      data=json.dumps({'sentence': sentence}))
    data = r.json()
    return data["data"]


def forward_paraphrase(sentence):
    # Making the request to the backend
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward", headers=headers,
                      data=json.dumps({'sentence': sentence}))
    data = r.json()
    return data["data"]


def visual_pos(doc, title: Optional[str] = "Dependency Parse & Part-of-speech tags"):
    if title:
        st.header(title)

    docs = [span.as_doc() for span in doc.sents]
    for sent in docs:
        html = displacy.render(sent, style="dep")
        html = html.replace("\n\n", "\n")
        st.write(get_svg(html), unsafe_allow_html=True)


if __name__ == "__main__":
    main()