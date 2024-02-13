import streamlit as st
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def chat():
    st.title("Legal Chat Assistant")

    model_options = {
        "GPT2_Custom_Contracts_124M_V1": "./fine_tuned_gpt2",
    }

    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), key="legal_assist_model_select")
    model_path = model_options[selected_model]
        
    temperature = st.sidebar.slider("Temperature", 0.5, 1.0, 0.7, 0.05, help="Controls the randomness of the predictions. Lower values make the model more confident, but less creative.")
    max_length = st.sidebar.slider("Max Length", 10, 500, 50, help="Controls the maximum length of the generated text. Longer lengths will take longer to generate.")
    num_beams = st.sidebar.slider("Num Beams", 1, 10, 5, help="Controls the number of beams to use in beam search. Higher values will generate more diverse text, but will take longer to generate.")
    no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 2, 5, 2, help="Controls the number of n-grams to prevent from repeating. Higher values will generate more diverse text, but will take longer to generate.")
    top_k = st.sidebar.slider("Top-k Sampling", 0, 50, 0, help="Controls the number of highest probability vocabulary tokens to keep for top-k sampling. Lower values will generate more diverse text, but will take longer to generate.")
    top_p = st.sidebar.slider("Top-p (Nucleus) Sampling", 0.0, 1.0, 0.9, help="Controls the cumulative probability of the highest probability vocabulary tokens to keep for nucleus sampling. Lower values will generate more diverse text, but will take longer to generate.")
    repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.2, help="Controls how much to penalize new tokens that are repeats of tokens already generated. Higher values will generate more diverse text, but will take longer to generate.")

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("How can I assist you?", key="unique_chat_input")
    if prompt is not None and prompt.strip() != "":
        if is_valid_input(prompt):
            cleaned_prompt = clean_input(prompt)

            with st.spinner('Getting response...'):
                try:
                    response = generate_legal_document(cleaned_prompt, model, tokenizer, max_length, temperature, num_beams, no_repeat_ngram_size, top_k, top_p, repetition_penalty)
                except RuntimeError as e:
                    response = "Sorry, there was an error generating a response. Please try a shorter prompt."

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message(name="assistant"):
                st.markdown(response)

def is_valid_input(input_text):
    
    return len(input_text) > 3 and any(char.isalpha() for char in input_text)

def clean_input(input_text):
    input_text = re.sub(r"(.)\1{2,}", r"\1", input_text)
    input_text = re.sub(r"[^a-zA-Z0-9\s.,!?']", '', input_text)
    input_text = input_text.strip()
    max_length = 500
    if len(input_text) > max_length:
        input_text = input_text[:max_length]
    return input_text

def generate_legal_document(prompt, model, tokenizer, max_length, temperature, num_beams, no_repeat_ngram_size, top_k, top_p, repetition_penalty):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoding = tokenizer.encode_plus(
        prompt, 
        return_tensors='pt',
        add_special_tokens=True, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length, 
            num_beams=num_beams, 
            no_repeat_ngram_size=no_repeat_ngram_size, 
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    prompt_pattern = re.escape(prompt)  
    generated_text = re.sub(f"^{prompt_pattern}", "", generated_text, flags=re.IGNORECASE).strip()

    return generated_text

if __name__ == "__main__":
    chat()
