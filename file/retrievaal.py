import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
from keras.models import load_model
import numpy
import tflearn
import tensorflow
import random
import json
import os

# nltk.download('punkt')

from settings import DATA_PATHH, CHECKPOINT_PATHH

# DATA_PATHH = '../data/all_data.json'
# CHECKPOINT_PATHH = "../model/re_training/model.h5"
# CHECKPOINT_DIRR = os.path.dirname(CHECKPOINT_PATHH)

with open(DATA_PATHH) as file:
    data1 = json.load(file)
words1 = list()
labels1 = list()
docs_x1 = list()
docs_y1 = list()
words = list()
labels = list()
docs_x = list()
docs_y = list()

for intent in data1['intents']:
    for pattern in intent['patterns']:
        wrds1 = nltk.word_tokenize(pattern)
        words1.extend(wrds1)
        docs_x1.append(wrds1)
        docs_y1.append(intent["tag"])

    if intent['tag'] not in labels1:
        labels1.append(intent['tag'])

words.extend(words1)
labels.extend(labels1)
docs_x.extend(docs_x1)
docs_y.extend(docs_y1)
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
out_empty = [0 for _ in range(len(labels))]
training = list()
output = list()
for x, doc in enumerate(docs_x):
    bag = list()
    wrds = [stemmer.stem(w.lower()) for w in doc]
    #     print(wrds)
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    # print(docs_y)
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)
training = numpy.array(training)
output = numpy.array(output)





tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
models = tflearn.DNN(net)
#

try:
    models.load(CHECKPOINT_PATHH)

except:
    history = models.fit(training, output, n_epoch=2000, batch_size=16, show_metric=True)
    models.save(CHECKPOINT_PATHH)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")

    while True:

        inp = input("Questions:  ")
        if inp.lower() == "quit":
            break

        results = models.predict([bag_of_words(inp, words)])
        print(results.shape)
        results_index = numpy.argmax(results)
        confidence = results[0][results_index]
        print("confidence : ", confidence)
        if confidence < 0.36 or confidence > 0.70:
            tag = labels[results_index]

            for tg in data1["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print (random.choice(responses))

        else:
            pass


# print(msg)
chat()