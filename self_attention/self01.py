# Self attention from scratch

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.layers import Softmax, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
input = np.array(["watch your back"])


def tokenize_input(input):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input)
    mapping = tokenizer.word_index
    input_tokenized_seq = tokenizer.texts_to_sequences(input)
    return input_tokenized_seq, mapping

res = tokenize_input(input)


def embedding_sequence(input, input_dimension:int, output_dimension:int, max_length:int):
    embedding_input = np.array(input)
    model = Sequential()
    model.add(Embedding(input_dim=input_dimension, output_dim=output_dimension, input_length=max_length))
    embedded_final_seq = model.predict(embedding_input)
    return embedded_final_seq

embed = embedding_sequence(res[0], 4,3,3)

X = torch.tensor(embed[0])
print(X)

w_q = nn.Linear(3,3, bias=False)
w_k = nn.Linear(3,3, bias=False)
w_v = nn.Linear(3,3, bias=False)


