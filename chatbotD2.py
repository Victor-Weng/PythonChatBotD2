import random
import json
from re import S
import requests #used (in this code) for reading discord messages 
import pickle
import numpy as np
import pyautogui as auto #enable keystrokes
import time
import sys #to stop code
import nltk
import math #Used to compare the similarity of two word lists
import re #used to clear out links and mentions
from collections import Counter #Used to compare the similarity of two word lists
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from better_profanity import profanity #used to censor swear words. Refer to https://mobilelabs.in/censor-bad-words-using-better-profanity-in-python/


from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

v = 'D2' #Version of file
directory = 'C:/Users/victo/OneDrive/Desktop/pythonCode/PythonChatbot' + v + '/' #Directory where the general files are found

a = np.random.random() #randomizer
cd = 1 #Cooldown between typing out messages
bfr = 1 #added cooldown between typing out messages, used as a multiplier of a, our brandomized value (between 0 and 1) 
ck = False #boolean for enabling the bot to type out messages
tck = False #boolean for enabling the bot to train through answer response format
nm = False #boolean for checking if there is a new message that is not from the bot
tempid = '0' #Temporarily holds the id of the last message that the bot responds to, to check if there is a new message in Discord
#### USED FOR TRAINING 
tempuser = '' #Temporarily holds the id of the last user to message
tempidt = '' #temporarily holds the id of the last message sent for TESTING 
tempresp = '' #Temporarily holds the response message of a user who sends multiple lines at the same time. 
tempstr = '' #Temporarily holds the string response message of a user who sends multiple lines at the same time, will be tokenized into ptrn
tempptrn = [] #Temporarily holds the pattern message of a user who sends multiple lines at the same time
# tempresp is a string while tempptrn is a list because tempresp needs to be appended to existing responses while ptrn doesnt. resp is converted to string for new patterns
tempbool = False #Temporary Boolean to Distinguish between response and pattern
tempptrnidentify = [] #temporarily holds the pattern that is similar to the one that is read
noo = 1 #New or Old. Boolean indicates whether the result should be appended to an existing pattern (old) or be made into a new one (new)
trainingbool = False #Boolean used to determine whether it is long-term training or short-term training
messagecounter = 0 #Couinter used to determine when to re-train again, discord loads up to 50 so we are setting the limit at 51
templist = [] #Temporarily holds nothing for the split function in counter_cosine_similarity function
templistsort = [] #Temporarily holds the probability values of the patterns

#counters used to summarize the amount of new and old patterns found
newptrncounter = 0
oldptrncounter = 0

################################################################### DISCORD VALUES ######################################################
auth = 'OTk4NzUxNTg2MjcwNTg0ODgy.GEkKzL.xQYPV3VhEo9HtcDlyV2wc9H3hiognKX6T5VMP4' #Authorization value, may need to be changed each time
#This is found by opening Discord on chrome, typing cntrl+shift+i to open dev tools, go into Network tab, type something in server, click "message" under "Name"
#then scrolling down to 'authorization' and copying the code
did = '763957300507181077' #'763957300507181077' < english hangout '1004869219457970256' < testing discord #Discord channel id #'
#This is found by refreshing the chrome page (still with cntrl+shift+i open). Filtering under 'Name' by 'messages'. Under 'Headers' on the right side, take the Request URL.
#You can ignore the '?limit=50'
uid = '998751586270584882' #user id of DualityOfMan#0643 (the bot)

###############################################################################################################################################


intents = json.loads(open(directory + 'intents' + v + '.json').read())
intentfilename = (directory + 'intents' + v + '.json')
words = pickle.load(open(directory + 'words' + v + '.pkl', 'rb'))
classes = pickle.load(open(directory + 'classes' + v + '.pkl', 'rb'))
model = load_model(directory + 'chatbotmodel' + v + '.h5')
ignore_letters = ['?', '!' , '.' , ',' , '"' , "'" , ';' , ':' , '(' , ')'] #characters to ignore
#ignore_words = [] #a "stop word list" to only keep key words for my intents.json pattern recognition. Starting with a simple one
#from https://www.ranks.nl/stopwords NOT IN USE AS I AM USING THE NATURAL LANGUAGE PROCESSING BUILT IN WORD_TOKENIZE
ignore_words = set(stopwords.words('english')) #stop words list from nltk to ignore when creating new patterns CURRENTLY TOO STRONG FOR CONVERSATIONS
ignore_words_small = ['i','a','about','an','are','as','at','be','by','for','from','how','in','is','it','of','on','or','that','the','this','to','was'
'what','when','where','will','with','the']


#functions to clean sentences, get bag of word, to predict class based on sentence, to get a response

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] #groups inflected forms of a word to be analyzed as a single item
    #found finds etc. are inflected forms, then analyzed as ==> find
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 #if the word matches, change its value in the array from 0 to 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) #create a bag of words from the sentence
    res = model.predict(np.array([bow]))[0] #predict result based on bag of words, take the first item ( [0] ) in the list
    ERROR_THRESHOLD = 0.35 #error threshold of 35%
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] #don't want too much uncertainty

    # lambda x: x[1] means "for every x cell (of lists), refer to the key set in the second position of that x list" 
    results.sort(key=lambda x: x[1], reverse=True) #sort by probability, reverse so that the highest probability is first
    #sort based on the second value in the lists [i, r] embedded within the results list
    #i is from 0 to 10 r is from the model.predict function that returns r which is the percentage given from the probability prediction
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if (len(intents_list) > 0):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
            else:
                result = 'That is a new one' #Input incase there is an error finding the tag. This seems to not work when the input is "Not sure what you mean"

        return result
    else: #random responses
        tempresultlist = ["You're asking the wrong guy","Not sure what you mean","Come again?","Try rephrasing that","Hmmmm.. idk","I really dont know, my capabilities are quite limited"]
        result = tempresultlist[random.randint(0,5)]
        return result

#Used with check_pattern to determine similarity between two word lists (existing and new)
def counter_cosine_similarity(c1, c2):
    global tempptrnidentify

    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    
    if(magA != 0 and magB != 0):
        tempptrnidentify = c1 #temporarily holds the existing pattern value

        return (dotprod / (magA * magB)) #returns a value between 0 and 1 that describes the similarity between the two word lists
    else:
        tempptrnidentify.clear()
        tempptrnidentify = ["this is a pattern dump for single-use and un-identifiable characters"]
        print("Cannot divide by 0") #This is likely caused by an existing pattern holding no value.
        return 1 

#a hard reset of variables for the training section
def hard_reset_training():
    global tempresp, tempstr, tempptrn, tempuser, tempidt, tempbool, tempptrnidentify, noo, tempid , templist, templistsort

    tempuser = '' #Temporarily holds the id of the last user to message
    tempidt = '' #temporarily holds the id of the last message sent for TESTING 
    tempresp = '' #Temporarily holds the response message of a user who sends multiple lines at the same time. 
    tempstr = '' #Temporarily holds the string response message of a user who sends multiple lines at the same time, will be tokenized into ptrn
    tempptrn = [] #Temporarily holds the pattern message of a user who sends multiple lines at the same time
    # tempresp is a string while tempptrn is a list because tempresp needs to be appended to existing responses while ptrn doesnt. resp is converted to string for new patterns
    tempbool = False #Temporary Boolean to Distinguish between response and pattern
    tempptrnidentify = [] #temporarily holds the pattern that is similar to the one that is read
    noo = 1 #New or Old. Boolean indicates whether the result should be appended to an existing pattern (old) or be made into a new one (new)
    templist = [] #Temporarily holds nothing for the split function in counter_cosine_similarity function
    templistsort = [] #Temporarily holds the probability values of the patterns 


def check_pattern(InputPatterns):

    global templist, templistsort, tempptrnidentify

    with open(intentfilename) as json_file:
        data = json.load(json_file)
        for value in data["intents"]:

            if(len(value['patterns']) == 1): #if there is only one value in the pattern, take the string value of it, split it into
                #seperate words, then reform a list
                tempstring = ' '.join(map(str, value['patterns'])) #Converts list to string
                templist = re.sub("[^\w]"," ",tempstring).split()
                counterJson = Counter(templist)   
            else:
                counterJson = Counter(value['patterns']) #Existing patterns
            TempInput = ' '.join(map(str, InputPatterns)) #Converts list to string
            InputPatternsSplit = TempInput.split()
        
            counterInput = Counter(InputPatternsSplit) #Input patterns to check
            sThreshold = 0.60 #minimum similarity result between the two word lists to be counted as an old pattern
            # print(counterJson)
            # print(counterInput)
            
            if(counter_cosine_similarity(counterJson, counterInput) > sThreshold):

                print(counter_cosine_similarity(counterJson, counterInput))

                if(len(value['patterns']) == 1):
                    templistsort.append([counter_cosine_similarity(counterJson, counterInput),tempstring]) #nested list with probabiltiy then the pattern
                else:
                    templistsort.append([counter_cosine_similarity(counterJson, counterInput),value['patterns']])

        if(len(templistsort) > 0): #if more than one has probability
            templistsort.sort(key=lambda x: x[0], reverse=True) #descending order for probability (which is found in the first position)

            print(templistsort)
            print(templistsort[0][1])

            result = 1 #Probability of new pattern being the same as an existing one is high, append the results
            if(isinstance(templistsort[0][1], str)): #if the value taken from above was a string, turn it into a list
                tempptrnidentify = templistsort[0][1].split("\n") #takes first value (highest probability) then the second value (the existing pattern) then splits it to a list from skipped lines
            

            else: #if the value taken from above was a list, keep it as a list and return it
                tempptrnidentify = templistsort[0][1]

 
            print("tempptrnidentify is : \n")
            print(tempptrnidentify)

        else:
            result = 0 #Probability of new pattern being the same as an existing one is low, make a new pattern + response in Json
        
        templist = [] #reset
        templistsort = []
        return result        
            


print('PROGRAM IS LAUNCHED')



##################################################### DISCORD INTEGRATION #################################################################
#TODO YOU MAY NEED TO UPDATED THE AUTHORIZATION KEY "auth" every time

#METHOD TO RETRIEVES MESSAGES FROM A DISCORD CHANNEL AND SERVER
def get_input(channelid): #needs input of chanel id

    headers = {
        'authorization': auth
    }
    r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages', headers=headers)
    json_parse = json.loads(r.text) #retrieve all of the messages r is defined from the request
    
    for value in json_parse:

        return(value['content']) #value gives all the information. Since it is a json file, you can add ['content'] to only obtain the message content

 #did is a string containing the channelid. It can be found in the same place as where requests are gotten up there (where channelid is replaced)


#METHOD TO CHECK IF THERE IS A NEW MESSAGE THAT IS SENT IN THE CHANNEL

def msg_check(channelid):

    global nm
    #Using Global Keyword so I can modify the global variable within the function (Note how tempid doesnt need to be re-initialized as it is not being modified locally)

    headers = {
        'authorization': auth
    }
    r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages', headers=headers)

    json_parse = json.loads(r.text) 
    
    for value in json_parse:
        if(value['id'] != tempid and value['author']['id'] != uid): #to check if there is a new message in Discord by comparing the new message id to that of the last one
            print('new')
            #used to determine when to reinitiate training
            nm = True
        else:
            nm = False
        break

#To reset msg check method
def msg_check_reset(channelid):

    global nm, tempid, uid

    headers = {
        'authorization': auth
    }
    r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages', headers=headers)

    json_parse = json.loads(r.text) 
    
    for g in json_parse:
        tempid = g['id']
        nm = False
        print("Reset Complete")
        break


#method to clean out links and extras
def clean_out(inputstring):

    inputstring = re.sub(r"(?:@|http?://|https?://|www)\S+", "", inputstring) #clean out links, mentions
    inputstring = re.sub(r"<.+?>","", inputstring)
    inputlist = ' '.join(inputstring.split()) #turns the input string into a list

    for i in range(len(inputlist)): #iterate through input list
        inputlist[i].replace("\\u","") #replace backslash u with nothing

    if len(inputlist) == 0: #INCASE ALL THAT WAS SENT WAS NOTHING, THIS IS WHAT WILL BE SAVED INTO THE PATTERNS SO THAT IT DOESNT RUIN THE counter_cosine_similarity function and divide by zero
        return "no input, sadly"

    return inputlist

#function to dump json file, completely rewrite
def write_json(data, filename):
    with open (filename, "w") as f:
        json.dump(data, f, indent=4)

# #function to append json file, add to the end
# def append_json(data, filename):
#     with open (filename, "a") as f:
#         json.dump(data, f, indent=1)


#To train the bot
def msg_train(channelid):
    global tempresp, tempstr, tempptrn, tempuser, tempidt, tempbool, noo, tempptrnidentify, tck, messagecounter, oldptrncounter, newptrncounter

    headers = {
        'authorization': auth
    }
    r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages', headers=headers)

    json_parse = json.loads(r.text) 
    
    for n in json_parse: # [1:] if you need to input something in chat to start the training to not register first message

        #if two DIFFERENT consecutive messages are from the SAME person 
        if (n['author']['id'] == tempuser and n['id'] != tempidt) or ('' == tempuser and n['id'] != tempidt):

            if tempbool == False:
                #append the two messages together under a different line, put the latest one (tempstr) below the older message (n['content'])
                
                tempresp = (n['content'] + "\n" + tempresp) 
                
                #used as a placeholder for the last Json dictionary value of user and id of message iterated through the loop
                tempuser = n['author']['id'] 
                tempidt = n['id']

            #TODO if boolean = true, set the new tempidt as the new author, have it fill the same with tempques this time
            elif tempbool == True:

                #append the two messages together under a different line, put the latest one (tempstr) below the older message (n['content'])
                
                tempstr = clean_out(n['content']) #TODO, CHANGE THIS SO THAT THE TEMPPTRN IS COMPLETE, AND THEN DETECT IF there is more than one 
                #element in the list, if there is, run clean_out on each individual element and have any objects with exactly
                #"no input, sadly", remove it. Same thing for responses.
                
                tempptrn.append(tempstr)

                #used as a placeholder for the last Json dictionary value of user and id of message iterated through the loop
                tempuser = n['author']['id'] 
                tempidt = n['id']

        #Once the messages are sent by another member
        else:

            if tempbool == False: #Just finished storing the response, move on to the input

                #Run this part again to initialize the response
                #tempstr = (n['content'] + "\n" + tempstr) 


                #CLEAN OUT MENTIONS AND LINKS
                tempstr = clean_out(n['content'])
                tempstr = profanity.censor(tempstr, 'o') #censoring out swear words, replacing each character of the swear word with 'o'

                tempptrn.append(tempstr)

                #used as a placeholder for the last Json dictionary value of user and id of message iterated through the loop
                tempuser = n['author']['id'] 
                tempidt = n['id']

                tempbool = True

            elif tempbool == True:

                        
                ###########################CLEAN UP RESPONSE#####################################
                #Response
                tempresp = profanity.censor(tempresp, 'o') #censoring out swear words, replacing each character of the swear word with 'o'
                tempresp = clean_out(tempresp)
            
                #############################################################################################

                noo = check_pattern(tempptrn) #New or old?  Boolean indicates whether the result should be appended to an existing pattern (old) or be made into a new one (new)



                ##THIS ENTIRE SECTION IS DEDICATED TO REMOVING ASCII'S BASICALLY, \U2342 OR UNICODE REPRESENTATIONS THAT JAM THE CODE

                if(type(tempptrnidentify) != list):
                    tempptrnidentifyNL = list(tempptrnidentify.items()) #transforms the counter to a list of tuples

                    for w in range(len(tempptrnidentifyNL)): #iterates through the list of tuples individually transforming each to a list (Overall, a nested list)
                        tempptrnidentifyNL[w] = list(tempptrnidentifyNL[w]) 

                    tempholderlist = tempptrnidentifyNL[0][0].split()
                else:
                    
                    tempptrnidentifyNL = list(tempptrnidentify)
                    tempholderlist = tempptrnidentify[0].split()

                if(not (tempholderlist[0]).isascii()):

                    print(tempptrnidentifyNL)

                    print("ascii identified, tempptrnidentifyNL SHOULD BE CLEARED")

                    tempptrnidentifyNL.clear()
                    tempptrnidentifyNL.append("this is a pattern dump for single-use and un-identifiable characters")

                tempholderlist = tempptrn[0].split()

                if(not (tempholderlist[0]).isascii()):
                            
                    print(tempptrn)
                    print("ascii identified, tempptrn SHOULD BE CLEARED")
                    
                    tempptrn.clear()
                    tempptrn.append("this is a pattern dump for single-use and un-identifiable characters")                


                if noo == 1: #OLD, Probability of new pattern being the same as an existing one is high, append the response
                
                    print("noo is equal to 1, test")
                    

                    if(tempresp != "no input, sadly"): #if it isnt an empty string
                    
                        
                        oldptrncounter += 1

                        #append tempresp
                        with open (intentfilename) as json_file:
                            data = json.load(json_file)
                            for value in data["intents"]: #iterate over the lists in data["intents"]
                                
                                if value["patterns"] == tempptrnidentifyNL:

                                    value["responses"].append(tempresp) #lists are mutable so append it this way
                                    print("succesfully appended in results")
                                    break
                        write_json(data, intentfilename)
                    else:
                        print("the new response was empty")
                    tempptrnidentify.clear()
                    tempptrnidentifyNL.clear()
                    

                #TODO else tempprtn becomes a new pattern with the corresponding tempresp as the response.
                elif noo == 0: #NEW Probability of new pattern being the same as an existing one is low, make a new pattern + response in Json
                    
                    if(tempresp != "no input, sadly"): #if it isnt an empty string
                        
                        print("new pattern")
                        newptrncounter += 1
            
                        for i in range (len(tempptrn)-1):
                            if (tempptrn[i] == "no input, sadly"):
                                del tempptrn[i]

                        if (len(tempptrn) > 0):
                            #make new pattern and response
                            with open (intentfilename) as json_file:
                                data = json.load(json_file)
                                temp = data["intents"]

                                print("Temp ptrn is: \n")
                                print(tempptrn)

                                tempresp = list(tempresp.split("\n")) #make tempresp a list for the new patterns

                                if(len(tempptrn) > 0):
                                    y = {"tag": "Training" + v + "v" + str(len(data["intents"]) + 100), "patterns": tempptrn, "responses": tempresp}
                                    temp.append(y)
                                    # +100 so that cleaning out intent tags with patterns and responses wont mess up the next wave after (As some tag numbers may repeat)

                            write_json(data, intentfilename) 
                        else:
                            print("the only new pattern(s) was 'no input, sadly'")
                    else:
                        print("the new response was empty")

                elif noo == 2:
                    print("Pattern skipped due to being null")
                 
                #Small reset on the variables that no longer need to be saved
                tempresp = ""
                tempuser = ""
                tempptrn = []
                tempbool = False
               

            #Reset on the temporary variables that are checked and restored at each stage
            tempidt = ""
            tempstr = ""



            #TODO Remove links, swearwords from the response part
            #TODO Remove useless words from the question part
            #TODO append both to the json file (find a way to make new variations of ONLY the json intents file if the bot stays the same?)
    print("Training is complete")
    print("New patterns: " + str(newptrncounter))
    print("Old patterns: " + str(oldptrncounter))
    
    #reset counters
    newptrncounter = 0
    oldptrncounter = 0

    if(trainingbool == False): #long-term training
        while(messagecounter < 51): #wait until 51 new messages have been sent so that they can all be retrained.
            
            msg_check(did)
            
            if(nm == True):
                messagecounter += 1
                print(messagecounter)
                msg_check_reset(did)

            time.sleep(0.2)
            
        else:
            print("message threshold, " + str(messagecounter) + " ," + "has been reached, training will restart...")
            messagecounter = 0 #reset the counter
            hard_reset_training() #hard resets the training variables
            tempbool = False
    elif(trainingbool == True): #short-term training
        sys.exit() #stops the bot
 


#if different, change boolean to false, fill temp ques with the next few lines running the same code. Perhaps embedd the 
            



#chatting/printing the messages

terminalmessage = input("Hello, welcome to the AI Chat bot. As a reminder, my answers are purely based on the messages I have been trained on, thus, not reflecting the" +
    " author, Victor's, personal opinion or values. To start using the bot, please input 'start' or the preset training modes. Enjoy :) \n" + "Please type something: \n") #to start training #TODO, FIX THIS

while True:

    message = get_input(did) #GET INPUT FROM DISCORD
    msg_check(did) #check if there is a new message and if it is not from the bot
    time.sleep(0.2)
    

    if(message == "Bye DualityOfMan"):
        sys.exit() #This fully stops the bot from running

    elif(message == "DualityOfMan"):
        ck = not ck #toggles the boolean to enable typing. "DualityOfMan" needs to be said by someone in the chat to enable the bot to begin typing
        tck = False #disables training as the bot would just train itself with its own question and responses

    elif(terminalmessage == "trainmodelong"):
        tck = not tck
        ck = False #disables typing as it is in training mode
        trainingbool = False #sets up for long term iterative training
    elif(terminalmessage == "trainmodeshort"):
        tck = not tck
        ck = False #disables typing as it is in training mode
        trainingbool = True #short-term one time training


    ints = predict_class(message)
    resp = get_response(ints, intents)

    if ck == True and nm == True: #if typing mode is on and there is a new message
        print("Input: " + message)
        print("Response: " + resp)
        msg_check_reset(did) #reset msg check


        time.sleep(cd) #cooldown between messages
        time.sleep(bfr*a) #buffer that is randomized to further delay each message
        for char in resp: #print out message
            auto.press(char)
            time.sleep(0.005) #time between typing each keystroke
        auto.press('enter') #enter the message

    elif tck == True: #if the bot is in training mode
        print("Currently in Training Mode, typing is disabled, new message detection is disabled")

        #run the method for training
        msg_train(did)




    elif nm == True: #if there is a new message
        print("Input: " + message)
        print("Response: " + resp)
        msg_check_reset(did) #reset msg check
        time.sleep(1) #time buffer

    else:
        print("awaiting new message...")
        time.sleep(1) #time buffer



#TODO
# 1. Perhaps go through discord Dms following response answer format? Maybe implement reactions under each message check, cross, and a refresh to indicate the following:
# check = good answer (keep) cross = bad answer (delete) refresh = (answer and question are flipped, flip them)
# 2. Using what you read from a pattern-response format (1 from a user, 1 response from a user), eliminate
# and get rid of useless words (as well as NSFW words) to find important key words. Also ignore links/files (find a way to identify)
# 3. If the pattern matches close enough (using our created functions) to an existing intents.json pattern, append the 
# response to the "responses": section.
# 4. Elif the response matches close enough (using our created functions) to an existing intents.json response, append the
# pattern to the "pattern": section.
# 5. If neither, create a new intent section with the "pattern": and "response": and continue. I HAVE ATTACHED A YT VIDEO ON THIS IN EASY
# 6. Make the json file rewrite itself as its scanning a Discord server chat
# 7. Afterwards, I can go through the intents briefly and remove/add necessary elements. Then, I can train the bot again.
# 8. Make sure it reads content that is considered "good"

# Extras: If someone says "Bye DualityOfMan", the bot closes its program
# Make a seperate method that is training mode. It will take all the 'content' from a user (multiple messages included, use if statements to see if its the same user)
# and then, it will record the 'content' in messages directly from the user that comes after as the response. It does not type anything during this period that is me


############## CREDITS and SOURCES :) ###################
# The fundamentals of the AI bot is made following https://www.youtube.com/watch?v=aBIGJeHRZLQ&ab_channel=KeithGalli and
# https://www.youtube.com/watch?v=1lwddP0KUEg&t=314s&ab_channel=NeuralNine YOUTUBE VIDEOS
# Using python requests from discord is used following https://www.youtube.com/watch?v=xh28F6f-Cds YOUTUBE VIDEO
# Appending and editing json files were used following the https://www.youtube.com/watch?v=9N6a-VLBa2I YOUTUBE VIDEO
# ALL OTHER ADDITIONS WERE DONE BY ME
# As well as various online users on StackExchange and other websites
# Friends and family for helping out with code and testing :0