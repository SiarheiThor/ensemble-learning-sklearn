import streamlit as st
from src import *
import joblib
from PIL import Image



# Use st.cache_resource to load and cache the model globally
@st.cache_resource
def load_model():
    model = joblib.load('trained_models/stacking_clf.joblib')
    return model

model = load_model()

# Streamlit app code
st.title('Digit Recognizer')

uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with Image.open(uploaded_file) as image:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Process the uploaded image directly
        preprocessed_digit = preprocess_image_for_digit_recognition(image)
    
    if preprocessed_digit is not None:
        preprocessed_image_display = preprocessed_digit.reshape(28, 28)
        st.image(preprocessed_image_display, caption='Preprocessed Image', width=150)
        
        # Ensure digit_image is reshaped correctly for a single sample
        digit_image_reshaped = preprocessed_digit.reshape(1, -1)

        # Predict the class of the digit
        prediction = model.predict(digit_image_reshaped)
        probability = model.predict_proba(digit_image_reshaped).max()
        
        # Display the prediction and probability
        st.write(f'Predicted Digit: {prediction[0]}')
        st.write(f'Probability: {probability:.4f}')
    else:
        st.write("Unable to locate a digit in the uploaded image.")


