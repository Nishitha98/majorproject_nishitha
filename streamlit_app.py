import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image as keras_image

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/diabetic_retinopathy_detection_project/Diabetic_Retinopathy_Detection.h5')
class_mapping={
    0:"Positive for diabetic retinopathy",
    1:"Negetive for diabetic retinopathy",
    #2:"Moderate",
    #3:"Severe",
}

def main():
    #st.title("image classifier")
    st.write("upload image for prediction")
# Function to preprocess the image
#def preprocess_img(img):
    #img=image.load_img(img,target_size=(224,224))
    #img = img.resize((224, 224))
    #img_array = np.asarray(img,dtype=np.float32)
    #img_array = np.expand_dims(img_array, axis=0)
    #img_array/=255.
    #img = preprocess_input(img)
    #img_array=img_to_array(img)
    #return img_array

# Function to make prediction
#def predict(img_array):
    #img = preprocess_img(img_array)
    #prediction = model.predict(img_array)
    #return prediction

# Streamlit app
st.title("Diabetic Retinopathy Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img=image.resize((224,224))
    img_array=keras_image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=img_array.astype('float32')/255.
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

# Button to make prediction
#if st.button('Predict'):
   # if uploaded_file is None:
        #st.warning("Please upload an image first.")
    #else:
    prediction = model.predict(img_array)
    predicted_class_index=np.argmax(prediction)
    predicted_class_name=class_mapping[predicted_class_index]
    st.write("prediction:")
    st.write(predicted_class_name)        #st.success("Prediction: {}".format(prediction))
if __name__=='__main__':
    main()
