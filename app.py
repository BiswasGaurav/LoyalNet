import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Loading the model
model = tf.keras.load_model('model.h5')

#Loading the Scalers and encoders
with open('label_encoder_gender.pkl', 'rb') as File:
    label_encoder_gender = pickle.load(File)
with open('onehot_encoder_geo.pkl', 'rb') as File:
    onehot_encoder_geo = pickle.load(File)
with open('scaler.pkl', 'rb') as File:
    scaler = pickle.load(File)
    