import streamlit as st
import requests
import tempfile
import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import random
import torch
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM ,AutoModelForSequenceClassification
import joblib
import chat
import spacy
import re
REPO_ID = "pile-of-law/legalbert-large-1.7M-2"

files_to_download = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "vocab.txt"
]

# Directory where to save the model
model_dir = "Models/legalbert-large-1.7M-2"

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Download each file
for file in files_to_download:
    file_path = os.path.join(model_dir, file)
    download_path = hf_hub_download(repo_id=REPO_ID, filename=file)
    with open(file_path, 'wb') as f:
        f.write(open(download_path, 'rb').read())


def extract_text_from_document(file):
    if file is not None:
        # Read the content of the file as bytes
        content_bytes = file.read()

        if content_bytes:
            # Decode the bytes into a string
            content = content_bytes.decode('utf-8')
            return content
        else:
            st.error("File is empty. Please choose a file with content.")
            return None
    else:
        return None
#________________________________________________*Classification code begins*__________________________________________
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    words_to_remove = ['Indian Kanoon']
    combined_pattern = re.compile('|'.join([url_pattern.pattern] + [re.escape(word) for word in words_to_remove]),
                                  flags=re.IGNORECASE)
    # Use the sub method to replace all matches with an empty string
    processed_text = re.sub(combined_pattern, '', text)
    return processed_text


def predict_legal_judgment(text, model_name):
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    # Load tokenizer and model based on the provided model name
    if model_name == "Indian-Legal-Bert":
        model = "law-ai/CustomInLawBERT"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
    elif model_name == "Indian-Custom-Bert":
        model = "brundamariswamy/Indian-Custom-Bert"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
    else:
        raise ValueError("Invalid model name")

    inputs = tokenizer(text[-512:], return_tensors="pt")

    with torch.no_grad():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to obtain class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # Get the predicted class label
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return probabilities, predicted_class

# Load the spaCy model
nlp = spacy.load("en_legal_ner_trf")
# Function to process text from a file
def process_text_from_file(text):
    # Read text from the file
    doc = nlp(text)
    # Create a dictionary to store entities by their names
    entity_dict = {}
    # Extract and store entities by their names
    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = set()
        # Lowercase the entity to make it case-insensitive
        entity = ent.text.lower()
        entity_dict[ent.label_].add(entity)

        # Remove duplicate entities within each category
    for label, entities in entity_dict.items():
        entity_dict[label] = list(entities)

    return entity_dict
#___________________________________________________________*Classification code ends*________________________________________________________


def generate_response_with_selected_model(model, tokenizer, input_tokenized):
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=150,
                                 max_length=250,
                                 early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    return summary


summary = "This is a summary of the document."

def main():
    inject_custom_css()

    st.warning(
        "This application is exclusively created for illustration and demonstration purposes. Please refrain from depending solely on the information furnished by the model.")

    tab1, tab2, tab3 = st.tabs(["Sum It Up!", "Classify This", "Say Hello!"])

    with tab1:
        st.title("Legal Document Summarizer")
        st.write("## Description")
        st.write("This app summarizes legal documents, highlighting key points and clauses.")
        st.write("## Steps")
        st.write("1. Upload a legal document.")
        st.write("2. Choose a summarization model.")
        st.write("3. Wait for the app to process the document.")
        st.write("4. View the summarized key points and clauses.")

        model_choice = st.sidebar.selectbox("Choose a Model", ["Pegasus Legal", "Pegasus Indian Legal","Indian-Legal-Bert", "Indian-Custom-Bert"])

        uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.write("Processing...")
            document_text = extract_text_from_document(uploaded_file)

            if document_text:
                if model_choice == "Pegasus Legal":
                    tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
                    model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
                    input_tokenized = tokenizer.encode(document_text, return_tensors='pt', max_length=1024, truncation=True)
                    summary = generate_response_with_selected_model(model, tokenizer, input_tokenized)
                elif model_choice == "Pegasus Indian Legal":
                    tokenizer = AutoTokenizer.from_pretrained("akhilm97/pegasus_indian_legal")
                    model = AutoModelForSeq2SeqLM.from_pretrained("akhilm97/pegasus_indian_legal")
                    input_tokenized = tokenizer.encode(document_text, return_tensors='pt', max_length=1024,
                                                       truncation=True)
                    summary = generate_response_with_selected_model(model, tokenizer, input_tokenized)
                else:
                    st.write(
                        "Please choose an appropriate model from one of the following options - BERT, GPT-3, or XLNet.")

                st.write("## Summary")
                st.write("Here's the summarized content of your document:")
                st.write(summary)
            else:
                st.write("Unable to process the document. Please try again with a different file format.")

    with tab2:
        st.title("Legal Case Judgement  Prediction and Extracting  Legal Named Entities")
        st.write("## Description")
        st.write("This advanced tool is designed to Predict Legal Judgements and Extract Legal Named Entities based on the provided content. It's useful for legal professionals who need quick insights to legal outcomes and the entities present in the case document.")
        st.write("## How It Works")
        st.write("1. **Upload a Legal Document:** Begin by uploading a document. Accepted formats include PDF, DOCX, and TXT.")
        st.write("2. **Choose Your Model:** Select from a range of AI models optimized for legal text analysis.")
        st.write("3. **Document Analysis:** The tool will Predict whether the appeals/claims filed by the appellant against the respondent is Accepted /Rejected and extract its Legal Named Entities from the document uploaded")
        st.write( "4. **Outcome Prediction:** Based on the analysis, it will also predict potential outcomes or implications.")

        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.write("Analyzing Document...")
            document_content = extract_text_from_document(uploaded_file)
            cleaned_text = remove_urls(document_content)
            words = cleaned_text.split()[-300:]
            st.write("Display last few lines from uploaded Legal case document:")
            st.write(' '.join(words))

            if cleaned_text:
                if model_choice in ["Indian-Legal-Bert", "Indian-Custom-Bert"]:
                    # Define the model name based on the user's choice
                    model_name = "Indian-Legal-Bert" if model_choice == "Indian-Legal-Bert" else "Indian-Custom-Bert"
                    # Get predictions and entities
                    probabilities, predicted_class = predict_legal_judgment(cleaned_text, model_name=model_name)
                    prediction_label = "Accepted" if predicted_class == 1 else "Rejected"
                    # Display results
                    st.write("## Analysis Results")
                    st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.8em;'>Legal Case Judgement:</span> {prediction_label}",unsafe_allow_html=True)
                    st.write("Prediction Confidence Bar Chart:")
                    st.bar_chart(probabilities[0].numpy(), use_container_width=True)

                    st.write("### Extracting Legal Named Entities:")
                    entities = process_text_from_file(cleaned_text)
                    for label, entities_list in entities.items():
                        st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.0em;'>{label}:</span> {', '.join(entities_list)}",unsafe_allow_html=True)
                else:
                    st.write("Please choose an appropriate model from one of the following options - BERT, RoBERTa, or Legal-BERT.")
            else:
                st.error("Unable to process the document. Please try a different format.")

    with tab3:
        st.write("## Connect with Us")
        with st.form("contact_form"):
            st.write("Feel free to reach out to us!")
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")
            submit_button = st.form_submit_button("Submit")

        st.write("### Socials")

        st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css">',
                    unsafe_allow_html=True)

        linkedin_icon = "<i class='fab fa-linkedin'></i>"
        github_icon = "<i class='fab fa-github'></i>"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Akhil Bharadwaj")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/akhil-bharadwaj-mab97/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/akhil97)", unsafe_allow_html=True)
        with col2:
            st.markdown("#### Brunda Mariswamy")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/brunda-mariswamy/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/bmariswamy5/)", unsafe_allow_html=True)
        with col3:
            st.markdown("#### Chirag Lakhanpal")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/chiraglakhanpal/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/ChiragLakhanpal)", unsafe_allow_html=True)


def inject_custom_css():
    custom_css = """
        <style>
            /* General styles */
            html, body {
                font-family: 'Avenir', sans-serif;
            }

            /* Specific styles for titles and headings */
            h1, h2, h3, h4, h5, h6, .title-class  {
                color: #C72C41; 
            }
            a {
                color: #FFFFFF;  
            } 
            /* Styles to make tabs equidistant */
            .stTabs [data-baseweb="tab-list"] {
                display: flex;
                justify-content: space-around; 
                width: 100%; 
            }

            /* Styles for individual tabs */
            .stTabs [data-baseweb="tab"] {
                flex-grow: 1; 
                display: flex;
                justify-content: center; 
                align-items: center; 
                height: 50px;
                white-space: pre-wrap;
                background-color: #C72C41; 
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
                font-size: 90px; 
            }

            /* Styles for the active tab to make it stand out */
            .stTabs [aria-selected="true"] {
                background-color: #EE4540 !important; 
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }
            /* Styles for the tab hover*/
            .stTabs [data-baseweb="tab"]:hover {
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }

        </style>    
    """
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == "__main__":

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a task", ["Summarize or Classify documents", "Legal Chat"])

    if page == "Summarize or Classify documents":

        main()
    elif page == "Legal Chat":
        chat.chat()
