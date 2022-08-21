import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer #makes it so words don't need to be hard coded, working, works worked, etc. are all the same

#Ignore warning errors, it is actually working 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

v = 'D2' #Version of file
directory = 'C:/Users/victo/OneDrive/Desktop/pythonCode/PythonChatbot' + v + '/' #Directory where the general files are found


intents = json.loads(open(directory + 'intents' + v + '.json').read())

words = [] #empty lists
classes = []
documents = []
ignore_letters = ['?', '!' , '.' , ',']  #characters to ignore
#ignore_letters = ['?', '!' , '.' , ',' , '"' , "'" , ';' , ':' , '(' , ')'] #characters to ignore

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #tokenize splits sentences into individual words
        words.extend(word_list) #add the tokenized words to the end of list 'words' extend = adding content to end of list and append = adding list to end of list
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #set eliminates duplicates, sorted turns back into list and sorts it

classes = sorted(set(classes))

pickle.dump(words, open(directory + 'words' + v + '.pkl', 'wb'))
pickle.dump(classes, open(directory + 'classes' + v + '.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training) #scramble training words
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


#Neural network model

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu')) #128 neurons in network that takes in an input of length based on our training model
model.add(Dropout(0.5)) #Dropout: Probability nodes randomly disconnected during training steps to stimulate a real environment
model.add(Dense(128, activation='relu')) # relu used for activation inputs
model.add(Dropout(0.5)) #THE VALUE INSIDE REPRESENTS THE PERCENTAGE OF NODES THAT ARE DROPPED. 0.5 ==> 50%
model.add(Dense(len(train_y[0]), activation='softmax')) 
# softmax for activation output, it returns an amount of activation values equal to the len(train_y[0])
# It takes in NUMBER OF OUTPUTS, activation type.

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Modile compile here, loss and accuracy are displayed
# can change loss to binary_crossentropy if looking to display 2 losses

hist = model.fit(np.array(train_x), np.array(train_y), epochs=400, batch_size=10, verbose=1)
# one epoch = when an entire dataset is passed through the computer/neural network once. It is often divided in to batches to
# pass it through the computer more easily.
# first two parameters are the input data

# Low accuracy => increase/change hyperparameters such as Hidden layers, neurons per layer, learning rate, epochs, optimizer, dropout layers, dropout values, batch-size
# High accuracy, slow run time => Can decrease these hyperparameters to increase effeciency
# The accuracy will plateau at a certain level if the training is complex


# The model.predict function is then used in our chatbot for incoming data.


model.save(directory + 'chatbotmodel' + v + '.h5', hist)

print("Done")

