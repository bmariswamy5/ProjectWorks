import streamlit as st
import utils
import Main
import base64
import os


def sentiment_main_page():
    st.markdown("<h1 style='font-size: 2 em; color: #000000; font-weight: bold;display: inline-block; text-align: right; white-space: nowrap;position: absolute; right: -150px;min-width: 600px;'> Sentiment Analysis and Entity Recognition of Legal Docuemnts</h1>", unsafe_allow_html=True)

def navigate_to_sentiment_main_page():
    sentiment_main_page()
def main():
    pages = {
        "Main Page": sentiment_main_page,
        "Next": Main.main
    }

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page's function
    pages[selected_page]()



if __name__ == "__main__":
    main()
