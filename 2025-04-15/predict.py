import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('/model_/catdog05040901_5_7.h5')

imgBy = 128
imgPath = '/workspace/2025-04-15/predict_.png'
img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(imgBy, imgBy))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0
x_new = np.array([img[0]])

preds = model.predict(x_new)
print(preds[:2])  # 예: [0.94, 0.01, 0.63, ...]S
#     1에 가까울수록 강아지 , 0에 가까울수록 고양이