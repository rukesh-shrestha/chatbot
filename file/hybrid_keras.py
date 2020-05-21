import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing, callbacks, utils

import pandas as pd
from gensim.models import Word2Vec
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
from settings import CHECKPOINT_DIR,CHECKPOINT_PATH,DATA_PATH,DATA_PATHH, CHECKPOINT_PATHH





# Include the epoch in the file name (uses `str.format`)
NUM_OF_BATCH = 32
NUM_OF_EPOCH = 100





dataset = pd.read_csv(DATA_PATH)
data1 = dataset.loc[:, ['human']]
data2 = dataset.loc[:, ['reply']]
questions = [sent_list[0] for sent_list in data1.values]
answers = [sent_list[0] for sent_list in data2.values]

answers_with_tags = list()
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)


answers = list()
for i in range(len(answers_with_tags)):
    answers.append('<START> ' + answers_with_tags[i] + ' <END>')


tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1




vocab = []
for word in tokenizer.word_index:
    vocab.append(word)


def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary


p = tokenize(questions + answers)
models = Word2Vec(p[0])

embedding_matrix = np.zeros((VOCAB_SIZE, 100))



# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
encoder_input_data = np.array(padded_questions)





# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(padded_answers)


# decoder_output_datatf.__version__
tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(onehot_answers)


encoder_inputs = tf.keras.layers.Input(shape=(None,))

encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)
print(output.shape)

models = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
models.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')






# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True, verbose=1
                                                 )

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
models.load_weights(latest)
#model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=8, epochs=5,callbacks=[cp_callback])
# Load the previously saved weights


def make_inference_models():
    encoder_models = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_models = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    print(decoder_state_input_h)
    type(decoder_state_input_h)

    return encoder_models, decoder_models





def str_to_tokens(sentence):
    words = sentence.lower().split()
    tokens_list = list()

    for word in words:
        if word in vocab:
            tokens_list.append(tokenizer.word_index[word])
        else:
            print("Pardon !!!")
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


def gene(take):
    enc_model, dec_model = make_inference_models()




    states_values = enc_model.predict(str_to_tokens(take))


    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)



model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open(DATA_PATHH).read())
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
            gene(inp)



chat()


