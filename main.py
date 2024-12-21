import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Load the trained TensorFlow model
def model_prediction(uploaded_image):
    try:
        cnn = tf.keras.models.load_model("my_model.h5")

        # Open the uploaded image using PIL and resize to 128x128
        image = Image.open(uploaded_image)
        image = image.resize((128, 128))  # Resize to match the model's input size

        # Convert image to array and normalize the pixel values
        input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Make predictions using the model
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)
        return result_index
    except UnidentifiedImageError:
        st.error("Unable to identify image file. Please upload a valid image.")
        return None

# Inject custom dark theme CSS styles
st.markdown("""
    <style>
        /* Body Styling */
        body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
            color: white;
            padding-top: 20px;
        }
        .sidebar .sidebar-title {
            color: #f5b242;
            font-size: 24px;
            font-weight: bold;
            text-transform: uppercase;
        }

        /* Header Styling */
        h1, h2, h3 {
            color: #f1f1f1;
        }

        /* Button Styling */
        .stButton button {
            background-color: #5cb85c;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
            font-size: 16px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #4cae4c;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        /* Image Styling */
        .stImage img {
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
            margin-top: 20px;
        }

        /* Text Content */
        .stMarkdown {
            font-size: 16px;
            color: #d1d1d1;
            line-height: 1.7;
        }

        /* Success message */
        .stSuccess {
            background-color: #2c6b2f;
            color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Input fields */
        .stTextInput, .stFileUploader {
            background-color: #2a2a2a;
            color: white;
            border: 1px solid #555;
            padding: 10px;
            border-radius: 8px;
        }
        .stTextInput input, .stFileUploader input {
            color: white;
        }

        /* Hover Effects */
        .stFileUploader:hover, .stTextInput:hover {
            border-color: #f5b242;
            background-color: #333;
        }

        /* Custom Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #4e4e4e;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page - Home
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # Update with the path to your homepage image
    st.image(image_path)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. 
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. 
    The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_image = st.file_uploader("Choose an Image:")

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image")

        if st.button("Predict"):
            result_index = model_prediction(uploaded_image)
            
            # If the image was valid and prediction is successful
            if result_index is not None:
                # List of class names corresponding to the trained model
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                st.success(f"Model predicts: {class_names[result_index]}")
