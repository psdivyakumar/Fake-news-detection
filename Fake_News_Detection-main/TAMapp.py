import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Function to load model
@st.cache_resource
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Public multilingual models
tokenizer1, model1 = get_model("bert-base-multilingual-cased")

tokenizer2, model2 = get_model("distilbert-base-multilingual-cased")

# UI
st.title("Tamil Fake News Detection")
st.subheader("Tamil and Multilingual News Classification")

user_input = st.text_area("Enter Tamil News Text")
button = st.button("Analyze")

# Labels
d = {
    1: 'Fake',
    0: 'True'
}

# Prediction
if user_input and button:

    # First model
    test_sample1 = tokenizer1(
        [user_input],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    output1 = model1(**test_sample1)

    y_pred1 = np.argmax(output1.logits.detach().numpy(), axis=1)

    probabilities1 = np.exp(output1.logits.detach().numpy()) / np.sum(
        np.exp(output1.logits.detach().numpy()),
        axis=1,
        keepdims=True
    )

    predicted_class_prob1 = probabilities1[0, y_pred1[0]]

    st.subheader("BERT Multilingual Result")
    st.write("Class Probability:", float(predicted_class_prob1))
    st.write("Prediction:", d[y_pred1[0]])

    st.markdown("---")

    # Second model
    test_sample2 = tokenizer2(
        [user_input],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    output2 = model2(**test_sample2)

    y_pred2 = np.argmax(output2.logits.detach().numpy(), axis=1)

    probabilities2 = np.exp(output2.logits.detach().numpy()) / np.sum(
        np.exp(output2.logits.detach().numpy()),
        axis=1,
        keepdims=True
    )

    predicted_class_prob2 = probabilities2[0, y_pred2[0]]

    st.subheader("DistilBERT Multilingual Result")
    st.write("Class Probability:", float(predicted_class_prob2))
    st.write("Prediction:", d[y_pred2[0]])
