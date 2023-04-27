import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2
import pickle
import plotly.graph_objs as go

feature_extraction_techniques = ['Color Histogram', 
                                 'CNN (Deep Learning)', 'HOG']

Model_selection = ['SVM', 'ANN']

#to preprocess the input image
image_array = ['initial garbarge value']


def preprocess_test_image(image):
  image_array = np.array(image.resize((300,300), resample=Image.BILINEAR)) #resizing and converting to a 2d array
  return image_array


#to extract features from the input image
test_data_features = ['initial garbage value']
def extract_test_features(image_array):
  hist = cv2.calcHist(image_array, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  hist = cv2.normalize(hist, hist).flatten()
  test_data_features = hist
  return test_data_features

#to predict
def prediction(test_input):
  loaded_model = pickle.load(open('svm.sav', 'rb'))
  test_data_prediction = loaded_model.predict(test_input)
  return test_data_prediction

def plot_color_hist_features(image):
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # Extract color histogram feature
  hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  hist = cv2.normalize(hist, hist).flatten()

  # Plot histogram
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  r, g, b = cv2.split(img)
  ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=img.reshape(-1,3)/255.)
  ax.set_xlabel('Red')
  ax.set_ylabel('Green')
  ax.set_zlabel('Blue')
  st.pyplot(fig)  


from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2

def plot_hog_features(image):
  # Convert image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Compute HOG features
  fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
  
  # Plot HOG features
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
  
  ax1.axis('off')
  ax1.imshow(image, cmap = plt.cm.gray)
  ax1.set_title('Input image')
  
  hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
  
  ax2.axis('off')
  ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
  ax2.set_title('Histogram of Oriented Gradients')
  
  st.pyplot(fig)