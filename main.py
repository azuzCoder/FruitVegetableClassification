import streamlit as st
from fastai.vision.all import *
import pandas as pd
import plotly.express as px
import pathlib

if platform.system() == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Fruit and vegetable model')

# explanation
st.write('Upload your image and find out if it is a fruit or a vegetable')

# uploading image
uploaded_image = st.file_uploader('Upload image', type=['jpeg', 'jpg', 'png'])

if uploaded_image:
    st.image(uploaded_image)

    # PIL convert
    img = PILImage.create(uploaded_image)

    # model
    model = load_learner('fruit_vegetable_model.pkl')

    # predictions
    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
