import nltk
import numpy
import tensorflow as tf
import tflearn
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        word, label, train, output = pickle.load(f)
except:
    word = []
    label = []
    doc_x = []
    doc_y = []

    for intent in data["intents"]:
        for pat in intent["patterns"]:
            wo = nltk.word_tokenize(pat)
            word.extend(wo)
            doc_x.append(wo)
            doc_y.append(intent["tag"])

            if intent["tag"] not in label:
                label.append(intent["tag"])

    word = [stemmer.stem(w.lower()) for w in word if w != "?"]
    word = sorted(list(set(word)))
    label = sorted(label)

    train = []
    output = []
    o_empty = [0 for _ in range(len(label))]

    for x, doc in enumerate(doc_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in word:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        out_row = o_empty[:]
        out_row[label.index(doc_y[x])] = 1
        train.append(bag)
        output.append(out_row)

    train = numpy.array(train)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((word, label, train, output), f)

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(train, output, n_epoch=1000, batch_size=9, show_metric=True)
    model.save("model.tflearn")

def bag_conatiner(s, word):
    con = [0 for _ in range(len(word))]
    s_word = nltk.word_tokenize(s)
    s_word = [stemmer.stem(wr.lower()) for wr in s_word]

    for se in s_word:
        for i, w in enumerate(word):
            if w == se:
                con[i] = 1

    return numpy.array(con)

def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        res = model.predict([bag_conatiner(inp, word)])[0]
        res_index = numpy.argmax(res)
        tag = label[res_index]

        if res[res_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    response = tg['responses']

            print(random.choice(response))

        else:
            print("I don't get that!! Please try again")
chat()
