import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
from configuration import DATA_PATHH
# nltk.download('punkt')
# nltk.download('wordnet')
model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open(DATA_PATHH,encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    # filter out predictions below a threshold
    # p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([bow(sentence, words)]))[0]
    ERROR_THRESHOLD = 0.50



    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)



    return_list = list()
    for r in results:
        # return_list.append((classes[r[0]], r[1]))
        rr = [classes[r[0]]],[r[1]]
        return_list.extend(rr)
        # print(type(return_list))
    return return_list





def chat():
    while True:
        inp = input("> - - - ")
        if inp.lower() == "quit":
            break
        results = predict_class(sentence=inp)

        results_index = np.array(results)
        print('results_index',results_index)
        confidence = results_index[1]
        # print(confidence)
        # print(type(results_index))
        co = (confidence.astype('float64'))
        print('co = ', co)
        print(type(co))
        val = np.float32(co)
        pyval = val.item()
        print(pyval)

        if pyval > 0.6:


            tag = results_index[0]

            list_of_intents = intents['intents']
            for i in list_of_intents:
                if (i['tag'] == tag):
                    result = random.choice(i['responses'])
                    break
            print (result)
        else:
            print('gene')



chat()





