import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
#from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import tensorflow_hub as hub
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
'''

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
#!pip install --upgrade tensorflow-hub
print("imported")

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
print(data.tail(10))

words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words)
tags = list(set(data["Tag"].values))
n_tags = len(tags)
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)
sentences = getter.sentences
max_len = 50
tag2idx = {t: i for i, t in enumerate(tags)}
X = [[w[0] for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X = new_X
print(new_X[15])
'''
from keras.preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
y[15]
'''
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tag2idx[w[2]] for w in s] for s in sentences]
from keras.preprocessing.sequence import pad_sequences
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
print(y[1])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=2018)
batch_size = 32
sess = tf.compat.v1.Session()
K.set_session(sess)

#import requests as req
#print(req.get("https://tfhub.dev/google/elmo/2", proxies={'https': 'https://10.208.73.221:7890'}).content)

#elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

model_dir="./elmo_2"
elmo_model = hub.Module(model_dir,trainable = True)

sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

'''
input = Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
'''
input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)


INIT_LR = 1e-3
EPOCHS = 3
#opt = adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = Model(input_text, out)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["categorical_accuracy"])
#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["sparse_categorical_accuracy"])
X_tr, X_val = X_tr[:1213*batch_size], X_tr[-135*batch_size:]
y_tr, y_val = y_tr[:1213*batch_size], y_tr[-135*batch_size:]
y_tr_a = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val_a = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
'''
a=np.array(X_tr)
b=np.array(X_val)
a=a.reshape(1)
b=b.reshape(1)
a=a.tolist()
b=b.tolist()
'''
print(np.array(X_tr).shape)
print(y_tr.shape)
print(np.array(X_val).shape)
print(y_val.shape)
history = model.fit(np.array(X_tr), y_tr_a, validation_data=(np.array(X_val), y_val_a),
                    batch_size=batch_size, epochs=EPOCHS, verbose=1)

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
X_te = X_te[:149*batch_size]
test_pred = model.predict(np.array(X_te), verbose=1)

idx2tag = {i: w for w, i in tags2index.items()}


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out


def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = test2label(y_te[:149 * 32])

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))

i = 390
p = model.predict(np.array(X_te[i:i+batch_size]))[0]
p = np.argmax(p, axis=-1)
print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
print("="*30)
for w, true, pred in zip(X_te[i], y_te[i], p):
    if w != "PADword":
        print("{:15}:{:5} ({})".format(w, tags[pred], tags[true]))

