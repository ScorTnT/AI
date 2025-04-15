import kagglehub

# Download latest version
path = kagglehub.dataset_download("salader/dogs-vs-cats")

print("Path to dataset files:", path)

import numpy as np
import os, glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# 이미지 크기 설정
imgBy = 128
imgCnt = 10000
epochs = 8
version = '7'
batch_size = 40
# 모델 구성: CNN 구조 정의
model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(imgBy, imgBy, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 셋 경로 설정 (train 폴더 기준)
base_dir = path
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 실제 폴더명이 'cats'와 'dogs'라면 그대로 사용, 만약 'cat'과 'dog'이면 수정 필요
cat_paths = sorted(glob.glob(os.path.join(train_dir, 'cats', '*.jpg')))
dog_paths = sorted(glob.glob(os.path.join(train_dir, 'dogs', '*.jpg')))

# 고양이, 강아지 각각 imgCnt 장씩만 사용 (총 2 * imgCnt장)
cat_subset = cat_paths[:imgCnt]
dog_subset = dog_paths[:imgCnt]

# DataFrame 생성: 파일 경로와 라벨 정보 기록
df_list = []
for fpath in cat_subset:
    df_list.append([fpath, 'cat'])
for fpath in dog_subset:
    df_list.append([fpath, 'dog'])
df = pd.DataFrame(df_list, columns=['filename', 'label'])

# train / validation 데이터 분할 (70:30)
df_train, df_val = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)

# ImageDataGenerator 설정 (rescale 적용)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

# train 데이터 제너레이터
train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='filename',
    y_col='label',
    class_mode='binary',
    target_size=(imgBy, imgBy),
    batch_size=batch_size,
    shuffle=True
)

# validation 데이터 제너레이터
validation_generator = datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='filename',
    y_col='label',
    class_mode='binary',
    target_size=(imgBy, imgBy),
    batch_size=batch_size,
    shuffle=True
)

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    verbose=1
)

# 모델 저장

model_dir = '/model_'
model.save(model_dir + '/catdog05040901_'+ version +'.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 모델 평가
print(version)
model = tf.keras.models.load_model(model_dir + '/catdog05040901_'+ version +'.h5')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.
)


test_generator = test_datagen.flow_from_directory(test_dir,
                                                  class_mode='binary',
                                                  target_size=(imgBy, imgBy))
test_loss, test_acc = model.evaluate(test_generator)
print('\n test 정확도:', test_acc)

