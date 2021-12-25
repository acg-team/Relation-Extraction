import os
import random
import pickle
import numpy as np
from utils import *
import pandas as pd
import networkx as nx
import tensorflow as tf
from tensorflow import keras
from node2vec import Node2Vec
from layers import AttentionLayer
from sklearn import preprocessing
from get_data_chemProt import get_data
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (GRU, Input, Dense, Conv1D, Flatten, Dropout, Embedding, concatenate, Bidirectional,
                                     MaxPooling1D, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D)
data_path = '../../data/ChemProt'
word_embedding_file = 'word_embedding_matrix.pkl'
data_Package_file = 'data_package.pkl'
model_file = 'model_ChemProtC'

embedding_dim = 100
pre_embedding_dim = 200
graph_emb_dim = 128
drop_out = 0.05
hidden_size = 64

# For training seed value should be selected randomly and consequent results should be averaged
# Setting seed value to ensure re-producible results:
seed_value = 94963
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def df2listss(df):
    full_sents, between_sents, label, srcs, trgs, src_types, trg_types = [], [], [], [], [], [], []
    for i in list(zip(df['text'], df['between_text'], df['label'], df['srcs'], df['trgs'], df['src_types'], df['trg_types'])):
        full_sents.append(i[0])
        between_sents.append(i[1])
        label.append(i[2])
        srcs.append(i[3])
        trgs.append(i[4])
        src_types.append(i[5])
        trg_types.append(i[6])
    return full_sents, between_sents, label, srcs, trgs, src_types, trg_types


def data_pre_process():
    le = preprocessing.LabelEncoder()
    train_path =  os.path.join(data_path, 'chemprot_training')
    dev_path =  os.path.join(data_path, 'chemprot_development')
    test_path =  os.path.join(data_path, 'chemprot_test_gs')

    train_df = get_data(os.path.join(train_path, 'chemprot_training_abstracts.tsv'), os.path.join(train_path, 'chemprot_training_relations.tsv'))
    dev_df = get_data(os.path.join(dev_path, 'chemprot_development_abstracts.tsv'), os.path.join(dev_path, 'chemprot_development_relations.tsv'))
    test_df = get_data(os.path.join(test_path, 'chemprot_test_abstracts_gs.tsv'), os.path.join(test_path, 'chemprot_test_relations_gs.tsv'))

    train_df = pd.concat([train_df, dev_df], ignore_index=True)

    train_text, train_between, train_label, train_srcs, train_trgs, train_src_types, train_trg_types = df2listss(train_df)
    dev_text, dev_between, dev_label, dev_srcs, dev_trgs, dev_src_types, dev_trg_types = df2listss(dev_df)
    test_text, test_between, test_label, test_srcs, test_trgs, test_src_types, test_trg_types = df2listss(test_df)

    ###############################################  GRAPH EMBEDDINGS  ################################################
    G = nx.DiGraph(directed=True)

    for triple in list(zip(train_srcs, train_label, train_trgs)):
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    node2vec = Node2Vec(G, dimensions=graph_emb_dim, walk_length=16, num_walks=100)
    model = node2vec.fit(window=10, min_count=1)


    node_size = len(G.nodes)
    graph_embedding_matrix = np.zeros((node_size+1, graph_emb_dim))
    for i, node in enumerate(G.nodes):
        graph_embedding_matrix[i] = model.wv[i]

    entity2id = {v:k for k, v in enumerate(G.nodes)}
    train_srcs_id = np.asarray([entity2id.get(x, node_size) for x in train_srcs])
    train_trgs_id = np.asarray([entity2id.get(x, node_size) for x in train_trgs])
    dev_srcs_id = np.asarray([entity2id.get(x, node_size) for x in dev_srcs])
    dev_trgs_id = np.asarray([entity2id.get(x, node_size) for x in dev_trgs])
    test_srcs_id = np.asarray([entity2id.get(x, node_size) for x in test_srcs])
    test_trgs_id = np.asarray([entity2id.get(x, node_size) for x in test_trgs])

    y_train = le.fit_transform(train_label)
    y_dev = le.transform(dev_label)
    y_test = le.transform(test_label)

    y_train = to_categorical(y_train)
    y_dev = to_categorical(y_dev)
    y_test = to_categorical(y_test)

    ##############################################  LOAD WORD TOKENIZER  ###############################################
    tokenizer = create_tokenizer(train_text)

    length = max_length(train_text)
    vocab_size = len(tokenizer.word_index) + 1
    trainX = encode_text(tokenizer, train_text, length)
    devX = encode_text(tokenizer, dev_text, length)
    testX = encode_text(tokenizer, test_text, length)

    ##############################################  LOAD TYPE TOKENIZER  ###############################################
    tokenizer_type = create_tokenizer(train_src_types + train_trg_types + test_src_types + test_trg_types)

    length_type = max_length(train_src_types + train_trg_types + test_src_types + test_trg_types)

    vocab_size_type = len(tokenizer_type.word_index) + 1

    trainX_src_type = encode_text(tokenizer_type, train_src_types, length_type)
    devX_src_type = encode_text(tokenizer_type, dev_src_types, length_type)
    testX_src_type = encode_text(tokenizer_type, test_src_types, length_type)

    trainX_trg_type = encode_text(tokenizer_type, train_trg_types, length_type)
    devX_trg_type = encode_text(tokenizer_type, dev_src_types, length_type)
    testX_trg_type = encode_text(tokenizer_type, test_trg_types, length_type)

    word_embedding_matrix = pickle.load(open(word_embedding_file, 'br'))

    return [graph_embedding_matrix, word_embedding_matrix, train_srcs_id, train_trgs_id, dev_srcs_id, dev_trgs_id,
            test_srcs_id, test_trgs_id, y_train, y_dev, y_test, vocab_size, node_size, trainX, devX, testX,
            vocab_size_type, length, length_type, le, trainX_src_type, devX_src_type, testX_src_type,
            trainX_trg_type, devX_trg_type, testX_trg_type]


def define_model(length, vocab_size, length_type, num_classes):
    inputs_tkn1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, pre_embedding_dim, weights=[word_embedding_matrix])(inputs_tkn1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(drop_out)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    cnn = Flatten()(pool1)

    inputs_tkn2 = Input(shape=(length, ))
    x1 = Embedding(vocab_size, pre_embedding_dim, weights = [word_embedding_matrix])(inputs_tkn2)
    x1 = SpatialDropout1D(drop_out)(x1)
    x1 = Bidirectional(GRU(hidden_size, return_sequences=True))(x1)
    att = AttentionLayer()(x1)
    avg_pool = GlobalAveragePooling1D()(x1)
    max_pool = GlobalMaxPooling1D()(x1)
    gru = concatenate([avg_pool, max_pool, att])

    inputs_src_type = Input(shape=(length_type, ))
    x1 = Embedding(vocab_size_type, embedding_dim)(inputs_src_type)
    x1 = SpatialDropout1D(drop_out)(x1)
    x1 = Bidirectional(GRU(hidden_size, return_sequences=True))(x1)
    max_pool_type_src = GlobalMaxPooling1D()(x1)

    inputs_trg_type = Input(shape=(length_type, ))
    x1 = Embedding(vocab_size_type, embedding_dim)(inputs_trg_type)
    x1 = SpatialDropout1D(drop_out)(x1)
    x1 = Bidirectional(GRU(hidden_size, return_sequences=True))(x1)
    max_pool_type_trg = GlobalMaxPooling1D()(x1)

    type = concatenate([max_pool_type_src, max_pool_type_trg])

    inputs_src_ontology = Input(shape=(1, ))
    x1 = Embedding(node_size +1, graph_emb_dim, weights = [graph_embedding_matrix])(inputs_src_ontology)
    x1 = SpatialDropout1D(drop_out)(x1)
    x1 = Bidirectional(GRU(hidden_size, return_sequences=True))(x1)
    max_pool_src = GlobalMaxPooling1D()(x1)

    inputs_trg_ontology = Input(shape=(1, ))
    x1 = Embedding(node_size +1, graph_emb_dim, weights = [graph_embedding_matrix])(inputs_trg_ontology)
    x1 = SpatialDropout1D(drop_out)(x1)
    x1 = Bidirectional(GRU(hidden_size, return_sequences=True))(x1)
    max_pool_trg = GlobalMaxPooling1D()(x1)

    ontology = concatenate([max_pool_src, max_pool_trg])

    merged = concatenate([cnn, gru, type, ontology])
    dense = Dense(64, activation='relu')(merged)
    outputs = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[inputs_tkn1, inputs_tkn2, inputs_src_type, inputs_trg_type, inputs_src_ontology, inputs_trg_ontology], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


##########################################################  RUN ########################################################
if __name__ == '__main__':
    if os.path.isfile(data_Package_file):
        data_Package = pickle.load(open(data_Package_file, 'br'))
    else:
        data_Package = data_pre_process()

        with open(data_Package_file, 'bw') as wr:
            pickle.dump(data_Package, wr)

    graph_embedding_matrix, word_embedding_matrix, train_srcs_id, train_trgs_id, dev_srcs_id, dev_trgs_id, \
    test_srcs_id, test_trgs_id, y_train, y_dev, y_test, vocab_size, node_size, trainX, devX, testX, vocab_size_type,\
    length, length_type, le, trainX_src_type, devX_src_type, testX_src_type, trainX_trg_type, \
    devX_trg_type, testX_trg_type = data_Package

    model = keras.models.load_model(model_file)
    y_pred = model.predict([testX, testX, testX_src_type, testX_trg_type, test_srcs_id, test_trgs_id])
    y_pred = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred, digits=3, target_names=le.classes_))