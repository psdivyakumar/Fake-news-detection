import base64
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Function to load model and tokenizer
@st.cache_resource
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load ALBERT model
tokenizer1, model1 = get_model("albert-base-v2")

# Load DistilBERT model
tokenizer2, model2 = get_model("distilbert-base-uncased")

# Background styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
right: 2rem;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("Fake News Detection")
st.subheader("English Fake News Detection using Transformer Models")

# Input box
user_input = st.text_area("Enter News Text to Analyze")

# Analyze button
button = st.button("Analyze")

# Label dictionary
d = {
    1: 'Fake',
    0: 'True'
}

# Prediction
if user_input and button:

    st.subheader("ALBERT Model Analysis")

    # ALBERT Prediction
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

    st.write("Class Probability:", float(predicted_class_prob1))
    st.write("Prediction:", d[y_pred1[0]])

    st.markdown("---")

    st.subheader("DistilBERT Model Analysis")

    # DistilBERT Prediction
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

    st.write("Class Probability:", float(predicted_class_prob2))
    st.write("Prediction:", d[y_pred2[0]])
