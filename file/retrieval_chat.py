import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from keras.models import load_model
import numpy
import tflearn
import tensorflow
import random
import os
from settings import DATA_PATHH, CHECKPOINT_PATHH









# save all of our data structures
# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open(DATA_PATHH, encoding="utf8") as json_data:
    intents = json.load(json_data)

MODEL_SAVE = '../model/model.tflearn'

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load(CHECKPOINT_PATHH)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))




def classify(sentence):
    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:

        # return_lishow_detaist.append((classes[r[0]], r[1]))
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return tuple of intent and probability
    return return_list


# def response(sentence, userID='123', show_details=False):
#     results = classify(sentence)
#     # if we have a classification then find the matching intent tag
#     if results:
#         # loop as long as there are matches to process
#         while results:
#             for i in intents['intents']:
#                 # find a tag matching the first result
#                 if i['tag'] == results[0][0]:
#                     # a random response from the intent
#                     return print(random.choice(i['responses']))
#
#             results.pop(0)
# #
# n = 'rukesh'
#
# for i in n:
#
#     response('hi')

import time



def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bow(inp, words)])
        print(results.shape)
        results_index = np.argmax(results)
        confidence = results[0][results_index]
        print("confidence : ", type(confidence))
        if confidence < 0.36 or confidence > 0.70:
            tag = classes[results_index]

            for tg in intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']


            print(random.choice(responses))


        else:
           print( 'Generative based chatbot')


chat()

