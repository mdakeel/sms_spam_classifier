import streamlit as st
import joblib
from src.components.data_transformation import DataTransformation
from src.utils.main_utils import MainUtils

#header
st.header("Email/SMS Spam Classifier", divider=True)

# Custom label with tight spacing
st.markdown("<h3 style='font-size:20px; margin-bottom:0; padding-bottom:0; color:#1F618D;'>✍️ Type Your Message</h3>", unsafe_allow_html=True)

# Text area without default label
txt = st.text_area(label='', placeholder="Type anything here...")

# Character count
st.markdown(f"🧮 You wrote **{len(txt)}** characters.")

#Logic

#loading model
utils = MainUtils()
preprocessor = utils.load_object('artifacts/preprocessor.pkl')
model = utils.load_object('artifacts/model.pkl')

#text transform
transform = DataTransformation()
transformed_text = [" ".join(transform.transform_text(msg)) for msg in [txt]] 

#vectorization
vectorize = preprocessor.transform(transformed_text)
predict = model.predict(vectorize)

#button
if st.button("Predict"):
    if txt.strip() == '':
        st.warning("Input cannot be empty.")
    elif predict == 1:
        st.error("Spam")    
    else:
        st.success("Not Spam")







