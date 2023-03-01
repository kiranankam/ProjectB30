import streamlit as st
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Attendance System using Facial Recognition",page_icon="memo",layout="wide")

img1 = Image.open('head.png')

st.header("Facial Recognition")


st.sidebar.image(img1)
st.sidebar.header("Predict Image")
img = st.sidebar.file_uploader("Choose Input Image",type=["jpg"])

def r2_score(y_true,y_pred):
    u = sum(square(y_true-y_pred))
    v = sum(square(y_true-mean(y_true)))
    return (1-u/(v+epsilon()))

model = load_model('network.h5')


img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img_array))

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    st.write(img_array.shape)

if img:
	img = Image.open(img)
	st.image(img,caption="Student Image")
	img_grey = img.convert("[L")
	img_grey = img_grey.resize((28,32))
	imgs = np.array(img_grey)
	data = np.reshape(imgs,(1,28,32,1))

	predic = model.predict(data)

	if st.sidebar.button("Student image"):
    		st.subheader("Image : {}".format(predic[0][0]))


