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

g_word_dict = {}


def os_listdir_ex(file_dir, find_name):  # 祖传代码
    result = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == find_name:
                result.append(os.path.join(root, file))
                # return result 测试用
    return result


def get_file_length(pFile):  # 得到文件长度,祖传代码
    fsize = os.path.getsize(pFile)
    return int(fsize)


def get_data_frame():  # 得到data frame
    # 得到webshell列表
    webshell_files = os_listdir_ex(".\\webshell", '.php')
    # 得到正常文件列表
    normal_files = os_listdir_ex(".\\noshell", '.php')
    label_webshell = []
    label_normal = []
    # 打上标注
    for i in range(0, len(webshell_files)):
        label_webshell.append(1)
    for i in range(0, len(normal_files)):
        label_normal.append(0)
    # 合并起来
    files_list = webshell_files + normal_files
    label_list = label_webshell + label_normal
    # 打乱数据,祖传代码
    state = np.random.get_state()
    np.random.shuffle(files_list)  # 训练集
    np.random.set_state(state)
    np.random.shuffle(label_list)  # 标签

    data_list = {'label': label_list, 'file': files_list}
    return pd.DataFrame(data_list, columns=['label', 'file'])


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
        results[i, sequence] = 1.
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
    '''
    word_bag = vectorize_sequences(word_bag)
    finished_bag = []
    for item in range(0, len(word_bag)):
        if len(word_bag[item]) != 0:
            for zeus in range(0, len(word_bag[item])):
                finished_bag.append(word_bag[item][zeus])
    print(len(finished_bag))
    '''
    '''
    tmp_fill = np.zeros((1337,), dtype=np.float)
    max_get_len = 1337
    if len(word_bag) < max_get_len:
        while True:
            word_bag = np.insert(word_bag, len(word_bag) - 1, tmp_fill, axis=0)
            if len(word_bag) >= max_get_len:
                break
    elif len(word_bag) > max_get_len:
        np.delete(word_bag, np.s_[max_get_len:])
    '''
    return word_bag


def build_network():
    global g_word_dict
    # y = label
    # 进来的file length_scaled entropy_scaled word_bag
    # 第一网络是一个TextCNN 词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    input_1 = keras.layers.Input(shape=(1337,), dtype='int16', name='word_bag')
    # 词嵌入（使用预训练的词向量）
    embed = keras.layers.Embedding(
        len(g_word_dict) + 1, 300, input_length=1337)(input_1)
    # 词窗大小分别为3,4,5
    cnn1 = keras.layers.Conv1D(
        256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = keras.layers.MaxPooling1D(pool_size=48)(cnn1)

    cnn2 = keras.layers.Conv1D(
        256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = keras.layers.MaxPooling1D(pool_size=47)(cnn2)

    cnn3 = keras.layers.Conv1D(
        256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = keras.layers.MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = keras.layers.Flatten()(cnn)
    drop = keras.layers.Dropout(0.2)(flat)
    model_1_output = keras.layers.Dense(
        1, activation='sigmoid', name='TextCNNoutPut')(drop)
    # 第一层好了
    # model_1 = keras.Model(inputs=input_1, outputs=model_1_output)

    # 第二层
    input_2 = keras.layers.Input(
        shape=(2,), dtype='float32', name='length_entropy')
    model_2 = keras.layers.Dense(
        128, input_shape=(2,), activation='relu')(input_2)
    model_2 = keras.layers.Dropout(0.4)(model_2)
    model_2 = keras.layers.Dense(64, activation='relu')(model_2)
    model_2 = keras.layers.Dropout(0.2)(model_2)
    model_2 = keras.layers.Dense(32, activation='relu')(model_2)
    model_2_output = keras.layers.Dense(
        1, activation='sigmoid', name='LengthEntropyOutPut')(model_2)
    # 第二层好了
    # 此时，我们将辅助输入数据与TextCNN层的输出连接起来,输入到模型中
    model_combined = keras.layers.concatenate([model_2_output, model_1_output])
    model_end = keras.layers.Dense(64, activation='relu')(model_combined)
    model_end = keras.layers.Dense(
        1, activation='sigmoid', name='main_output')(model_end)

    # 定义这个具有两个输入和输出的模型
    model_end = keras.Model(inputs=[input_2, input_1],
                            outputs=model_end)
    model_end.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
    '''
    # 第二层就是自己写的128 - drop 0.4 - 64 - drop 0.2 - 32 简单地分类,2输入 length_scaled entropy_scaled
    model_2 = keras.Sequential()
    model_2.add(keras.layers.Dense(
        128, input_shape=(2,), activation='relu'))
    model_2.add(keras.layers.Dropout(0.4))
    model_2.add(keras.layers.Dense(
        64, activation='relu'))
    model_2.add(keras.layers.Dropout(0.2))
    model_2.add(keras.layers.Dense(
        32, activation='relu'))
    # 第二网络好了
    model_2.add(keras.layers.Dense(1, activation='sigmoid'))

    # 融在一起
    # model_1.summary()
    model_combined = keras.layers.concatenate([model_2.output, model_1.output])

    # 融在一起
    model_end = keras.layers.Dense(
        2, activation="relu")(model_combined)
    model_end = keras.layers.Dense(1, activation="sigmoid")(model_end)
    model_end = keras.Model(
        inputs=[model_2.input, model_1.input], outputs=model_end)
    model_end.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy', 'loss'])
    # model_end = keras.layers.Dense(2, activation="relu")(model_combined)
    # model_end = keras.layers.Dense(1, activation="sigmod")(model_end)
    # model_end = keras.Model(inputs=[x.input, y.input], outputs=z)
    '''
    return model_end


# get_functions("C:\\Users\\Administrator\\Desktop\\webshell检测\\webshell\\一句话\\一句话.php")
data_frame = get_data_frame()
data_frame['length'] = data_frame['file'].map(
    lambda file_name: get_file_length(file_name)).astype(int)
data_frame['entropy'] = data_frame['file'].map(
    lambda file_name: get_file_entropy(file_name)).astype(float)
# 归一化这两个东西
scaler = StandardScaler()
data_frame['length_scaled'] = scaler.fit_transform(
    data_frame['length'].values.reshape(-1, 1), scaler.fit(data_frame['length'].values.reshape(-1, 1)))
data_frame['entropy_scaled'] = scaler.fit_transform(
    data_frame['entropy'].values.reshape(-1, 1), scaler.fit(data_frame['entropy'].values.reshape(-1, 1)))
# 导入词袋
data_frame['word_bag'] = data_frame['file'].map(
    lambda file_name: get_file_word_bag(file_name))

data_train_pre = data_frame.filter(
    items=['length_scaled', 'entropy_scaled'])
data_train_y = tf.constant(data_frame.filter(
    items=['label']))
data_train_x_1 = tf.constant(data_train_pre)
data_train_x_2 = tf.constant(
    vectorize_sequences(data_frame['word_bag'].values))
# 现在这个是一个 (batch_size,(1337个单词[1337个hot code]))
network_model = build_network()
network_model.summary()
history = network_model.fit(
    x=[data_train_x_1, data_train_x_2], y=data_train_y, batch_size=128, epochs=128)
network_model.save('huoji.h5')
