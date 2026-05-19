import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_data()
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load the first model
tokenizer1, model1 = get_model("Shivanikumar/Final-Tamilbert")

# Load the second model
tokenizer2, model2 = get_model("Shivanikumar/Multilingual-Distil")

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    1: 'Fake',
    0: 'True'
}

if user_input and button:
    # Analyze with the first model
    test_sample1 = tokenizer1([user_input], padding=True, truncation=True, max_length=20, return_tensors='pt')
    output1 = model1(**test_sample1)
    y_pred1 = np.argmax(output1.logits.detach().numpy(), axis=1)
    probabilities1 = np.exp(output1.logits.detach().numpy()) / np.sum(np.exp(output1.logits.detach().numpy()), axis=1, keepdims=True)
    predicted_class_prob1 = probabilities1[0, y_pred1[0]]
    
    st.write("Indic-Bert:")
    st.write("Class Probability: ", predicted_class_prob1)
    st.write("Indic-Bert Prediction: ", d[y_pred1[0]])
    
    # Analyze with the second model
    test_sample2 = tokenizer2([user_input], padding=True, truncation=True, max_length=20, return_tensors='pt')
    output2 = model2(**test_sample2)
    y_pred2 = np.argmax(output2.logits.detach().numpy(), axis=1)
    probabilities2 = np.exp(output2.logits.detach().numpy()) / np.sum(np.exp(output2.logits.detach().numpy()), axis=1, keepdims=True)
    predicted_class_prob2 = probabilities2[0, y_pred2[0]]
    
    st.write("Distil-mbert:")
    st.write("Class Probability: ", predicted_class_prob2)
    st.write("Distil-mbert Prediction: ", d[y_pred2[0]])
