# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 22:29:56 2025

@author: DIPUNI SATHUA
"""
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import tensorflow as tf

html_temp="""
    <div style = "background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">X-ray Image Classifier</h2>
    </div>
    """
    
st.markdown(html_temp,unsafe_allow_html = True)
img_size = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

from tensorflow.keras.models import load_model
model = load_model("model_10.h5")
print("model loaded")
def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader(label=" ", type=['jpeg', 'jpg', 'png'])

    if file is not None:
        # Load as grayscale
        img = tf.keras.preprocessing.image.load_img(file, target_size=(img_size, img_size), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape(-1, img_size, img_size, 1)  # batch dimension

        st.image(file, caption="Uploaded Image", use_container_width=True)
        st.write("")
        st.write("")

        if st.button("PREDICT"):
            # Make prediction
            prediction = model.predict(img_array / 255.0)  # normalize
            print(prediction)
            print(round(prediction[0][0]))
            # Determine predicted class and percentage
            if round(prediction[0][0]) == 1:
                preds = f"{CATEGORIES[1]} - {prediction[0][0]*100:.2f}%"
            else:
                preds = f"{CATEGORIES[0]} - {(1 - prediction[0][0])*100:.2f}%"

            st.write(preds)



def main():
    load_classifier()


if __name__ == "__main__":
	main()


