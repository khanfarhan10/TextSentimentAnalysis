
import streamlit as st
import requests
import json


def main():
    # Applying styles to the buttons
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

    # Heading
    st.header("Text Sentiment Analysis")

    # Text area for user input
    user_input = st.text_area("Enter your text here", "")

    if st.button("Analyze"):
        with st.spinner('Analyze Text'):
            output = forward(user_input)
            print(output)


def forward(sentence):
    # Making the request to the backend
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward", headers=headers,
                      data=json.dumps({'sentence': sentence}))
    data = r.json()
    return data["data"]


if __name__ == "__main__":
    main()