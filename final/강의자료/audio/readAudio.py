
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

audio_result = []

audio_path = '/workspace/final/audio/011803.wav'
# audio_path = '/workspace/final/audio/007527.mp3'
audio_sample, sampling_rate = librosa.load(audio_path, sr=None)

# print(audio_sample)
# print(len(audio_sample))
# print(sampling_rate)
# print(len(audio_sample) / sampling_rate)
np_audio_sample = np.array(audio_sample)
#print(np_audio_sample)

feature1 = np.mean(np.abs(audio_sample))    #(1) 각 레벨 안에 있는 모든 계수들에 대한 절 대값의 평균값
feature2 = np.mean(audio_sample ** 2)       #(2) 각 레벨 안에 있는 모든 계수들을 제곱하여 구한 평균값
feature3 = np.std(audio_sample)             #(3) 각 레벨 안에 있는 모든 계수들의 표준편차
feature4 = np.median(audio_sample)          #(4) 각 레벨 안에 있는 모든 계수들의 중앙값

audio_sample2 = audio_sample ** 2
audio_result.append([feature1, feature2, feature3, feature4])
print(audio_result)

plt.plot(np.array(audio_sample),'black')
plt.show()