# coding=utf-8
import fileinput
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
import re
import nltk
import sklearn
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as keras_preprocessing
from sklearn.preprocessing import StandardScaler
import chardet
import math
from joblib import load
g_word_dict = {}
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def os_listdir_ex(file_dir, find_name):  # 祖传代码
    result = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == find_name:
                result.append(os.path.join(root, file))
                # return result  # 测试用
    return result


def get_file_length(pFile):  # 得到文件长度,祖传代码
    fsize = os.path.getsize(pFile)
    return int(fsize)


def flush_file(pFile):  # 清洗php注释
    file = open(pFile, 'r', encoding='gb18030', errors='ignore')
    read_string = file.read()
    file.close()
    m = re.compile(r'/\*.*?\*/', re.S)
    result = re.sub(m, '', read_string)
    m = re.compile(r'//.*')
    result = re.sub(m, '', result)
    m = re.compile(r'#.*')
    result = re.sub(m, '', result)
    return result


# 得到文件熵 https://blog.csdn.net/jliang3/article/details/88359063
def get_file_entropy(pFile):
    clean_string = flush_file(pFile)
    text_list = {}
    _sum = 0
    result = 0
    for word_iter in clean_string:
        if word_iter != '\n' and word_iter != ' ':
            if word_iter not in text_list.keys():
                text_list[word_iter] = 1
            else:
                text_list[word_iter] = text_list[word_iter] + 1
    for index in text_list.keys():
        _sum = _sum + text_list[index]
    for index in text_list.keys():
        result = result - float(text_list[index])/_sum * \
            math.log(float(text_list[index])/_sum, 2)
    return result


def vectorize_sequences(sequences, dimention=1337):
    # 创建一个大小为（25000，10000）的全零矩阵
    results = np.zeros((len(sequences), dimention))
    for i, sequence in enumerate(sequences):
        if i > dimention:
            break
        try:
            results[i, sequence] = 1.
        except:
            break

    return results


def get_file_word_bag(pFile):
    global g_word_dict
    english_punctuations = [',', '.', ':', ';', '?',
                            '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', 'php', '<', '>', '\'']
    clean_string = flush_file(pFile)
    word_list = nltk.word_tokenize(clean_string)
    # 过滤掉不干净的
    word_list = [
        word_iter for word_iter in word_list if word_iter not in english_punctuations]

    keras_token = keras.preprocessing.text.Tokenizer()  # 初始化标注器
    keras_token.fit_on_texts(word_list)  # 学习出文本的字典
    g_word_dict.update(keras_token.word_index)
    # 通过texts_to_sequences 这个dict可以将每个string的每个词转成数字
    sequences_data = keras_token.texts_to_sequences(word_list)
    # 将每条文本的长度设置一个固定值, ps 超过1337个字符的"单词"不用说肯定是某个骇客想把大马变免杀马
    # word_bag = keras_preprocessing.sequence.pad_sequences(sequences_data, maxlen=1337, dtype='int16')
    word_bag = []
    for index in range(0, len(sequences_data)):
        if len(sequences_data[index]) != 0:
            for zeus in range(0, len(sequences_data[index])):
                word_bag.append(sequences_data[index][zeus])
    return word_bag


file_path = '.\\1.php'

entropy = get_file_entropy(file_path)
length = get_file_length(file_path)
word_bag = get_file_word_bag(file_path)
array_input = np.array([[entropy, length]])

data_frame = pd.DataFrame(
    {'length': [length], 'entropy': [entropy], 'word_bag': [word_bag]}, columns=['length', 'entropy', 'word_bag'])
# scaler = StandardScaler()
scaler_entropy = load('scaler_entropy.joblib')
scaler_length = load('scaler_length.joblib')
data_frame['length_scaled'] = scaler_length.transform(
    data_frame['length'].values.reshape(-1, 1))
data_frame['entropy_scaled'] = scaler_entropy.transform(
    data_frame['entropy'].values.reshape(-1, 1))

data_train_pre = data_frame.filter(items=['length_scaled', 'entropy_scaled'])
# data_train_pre = data_frame.filter(items=['length', 'entropy'])
data_train_x_1 = tf.constant(data_train_pre)
data_train_x_2 = tf.constant(
    vectorize_sequences(data_frame['word_bag'].values))
print(data_frame.head())

model_name = 'huoji1.h5'  # huoji.h5 huoji_scaled.h5 huoji_no_scale.h5
model = keras.models.load_model(model_name)
model.summary()
print(data_train_x_1, data_train_x_2)
prediction = model.predict([data_train_x_1, data_train_x_2])
print(prediction)
