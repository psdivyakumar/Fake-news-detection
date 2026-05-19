import base64
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# @st.experimental_user
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64("image.jpg")

@st.cache_data()
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load the first model
tokenizer1, model1 = get_model("Shivanikumar/Albert-base-v2")

# Load the second model
tokenizer2, model2 = get_model("Shivanikumar/Distilbert-base")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

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
    
    # ... other analysis and prediction logic ...
    
    st.write("Albert model:")
    st.write("Class Probability: ", predicted_class_prob1)
    st.write("Albert Prediction: ", d[y_pred1[0]])
    
    # Analyze with the second model
    test_sample2 = tokenizer2([user_input], padding=True, truncation=True, max_length=20, return_tensors='pt')
    output2 = model2(**test_sample2)
    y_pred2 = np.argmax(output2.logits.detach().numpy(), axis=1)
    probabilities2 = np.exp(output2.logits.detach().numpy()) / np.sum(np.exp(output2.logits.detach().numpy()), axis=1, keepdims=True)
    predicted_class_prob2 = probabilities2[0, y_pred2[0]]
    
    st.write("Distilbert:")
    st.write("Class Probability: ", predicted_class_prob2)
    st.write("Distilbert Prediction: ", d[y_pred2[0]])
