# Self attention from scratch

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.layers import Softmax, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
input = np.array(["aloo kkha loo"])


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


def attention_score_and_scaling(input_embed, dim_k:int):
    X = torch.tensor(input_embed)
    w_q = nn.Linear(3,3, bias=False)
    w_k = nn.Linear(3,3, bias=False)
    w_v = nn.Linear(3,3, bias=False)

    WQ_tensor = w_q.weight.data
    WK_tensor = w_k.weight.data
    WV_tensor = w_v.weight.data

    query_vector = torch.matmul(X, WQ_tensor)
    key_vector = torch.matmul(X, WK_tensor)
    value_vector = torch.matmul(X, WV_tensor)

    attention_score = torch.matmul(query_vector, key_vector)/dim_k

    contextual_attention_embeddings = torch.softmax(attention_score, dim=-1)
    print(contextual_attention_embeddings)

attention_score_and_scaling(embed[0], 3)
