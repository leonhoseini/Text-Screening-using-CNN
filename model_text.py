from os import listdir, path
from stop_words import get_stop_words as exclude
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pprint import pprint
tokenizer=Tokenizer()

def gen_x(xtext,tokenizer,max_len=None,for_train=False,):
    if for_train:
        tokenizer.fit_on_texts(xtext)
    encoded_xtext=tokenizer.texts_to_sequences(xtext)
    if not max_len:
        max_len=max([len(s.split()) for s in xtext])
    train_x=pad_sequences(encoded_xtext,maxlen=max_len,padding='post')
    if for_train:
        return train_x,max_len
    return train_x
def cleanup(w,clean_sw=True):
    w=w.strip().lower()
    if not w.isalpha():
        return None
    if clean_sw and w in exclude('english'):
        return None
    if len(w)==1:
        return None
    return w
        
def clean(data, clean_sw):
    out=[]
    for doc in data:
        wout=[]
        for w in doc.split():
            w=cleanup(w,clean_sw)    
            if w==None:
                continue
            wout.append(w)
        out.append(' '.join(wout))
    return out
def get_data(d='txt_sentoken',do_clean=True,filter_stops=True):
    
    train_x=[];train_y=[];test_x=[];test_y=[]
    
    for p in ['neg','pos']:
        for filename in listdir(path.join(d,p)):
            dfile=path.join(d,p,filename)
            data=open(dfile).read()
            train_x.append(data)
    if do_clean:
        ct=clean(train_x,filter_stops)
    else:
        ct=train_x
    
    l=1000
    trainl=int(l*0.90);testl=int(l*0.10)
    train_x_neg=ct[0:trainl] ; train_x_pos=ct[l:l+trainl]
    train_y_neg=[0 for i in range(len(train_x_neg))]
    train_y_pos=[1 for i in range(len(train_x_pos))]
    train_x=train_x_neg+train_x_pos
    train_y=train_y_neg+train_y_pos
    
    test_x_neg=ct[trainl:l];test_x_pos=ct[l+trainl:]
    test_y_neg=[0 for i in range(len(test_x_neg))]
    test_y_pos=[1 for i in range(len(test_x_pos))]
    test_x=test_x_neg+test_x_pos
    test_y=test_y_neg+test_y_pos
    #tokenizer=Tokenizer()
    input_train_x=train_x
    train_x,max_len=gen_x(train_x,tokenizer,for_train=True)
    test_x=gen_x(test_x,tokenizer,max_len=max_len)
    
    pprint(input_train_x[0][:50])
    pprint(train_x[0][:9])
    for w in input_train_x[0][:50].replace(':','').split():
        print(w,'=',tokenizer.word_index[w])
    print()
    
    inputs=len(tokenizer.word_index)+1
    print(inputs)
    return train_x,train_y,test_x,test_y,inputs,max_len,tokenizer
    
if __name__ == "__main__":
    train_x,train_y,test_x,test_y,inputs,max_len,t=get_data()
    #get_data()
    print('x[0]',train_x[0])
    print('y[0]',train_y[0])
    
    
#####
from keras.models import Sequential
from keras.layers import Dense, Conv1D,Dense,Flatten,MaxPooling1D,Embedding,Activation
import numpy
import os
import sys
from numpy import loadtxt
from keras.models import load_model
from numpy import *
from pandas import *
import pickle
def txt_cnn(inputs,max_length,dim=20):
    model=Sequential()
    model.add(Embedding(inputs,dim,input_length=max_length))
    model.add(Conv1D(filters=32,kernel_size=5,activation='relu',padding='valid'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model
conf={'default': dict(model=txt_cnn)}
def train(name, train_x,train_y,epochs,batches,inputs,max_length):
    mparams=conf[name]
    model=mparams['model']
    model=model(inputs,max_length)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy"])
    model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=epochs,batch_size=batches,verbose=2)
    scores=model.evaluate(train_x,train_y, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save("model.h5")
    print("Saved model to disk")
    return model, name, mparams
if __name__ == '__main__':
    # Getting our command line parameters
    #name, epochs, batches=get_params()
    train_x, train_y, test_x, test_y, inputs, max_length, t=get_data(do_clean=True, filter_stops=True)
    print('Train/Test Data lenght', len(train_x), len(test_x))
    model, name, mp =train('default', train_x, train_y, 5, 2, inputs, max_length)
    
    epochs=4;batches=4
    mname='model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname+'.h5')
    with open(mname+'-tokenizer.pickle', 'wb') as ts:
        pickle.dump(t, ts)
    title='%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print('Evaluation for %s' % title)
    loss, acc = model.evaluate(train_x, train_y, verbose=2)
    print('Train Accuracy: %.2f%%' % (acc*100))
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    print('Test Accuracy: %.2f%%' % (acc*100))

if __name__ == '__main__':
    
    # Get the tweet.
    print("Type in one tweet per line and hit CTRT-D when you're done:")

    testt=open('testt.txt','w')
    L=input('leave your tweet: ') 
    testt.writelines(L)
    testt.close()
    testt=open('testt.txt')
    # testt='I hate that movie. Awkwerd. I cursed the friend for inviting this boring movie.'
    for tweet in testt:
        # Cleanup the tweet before we use our model.
        t=clean([tweet], True)
        # Encode and pad our tweet with the same tokenizer
        # that we've used for training and testing.
        # We've set our own variable in
        # tokenizer._max_padding_len on training to store
        # informations about the maximum lenght of our encoded text.
        t=tokenizer.texts_to_sequences(t)
        t=pad_sequences(t, maxlen=max_len, padding='post')
        # Get one of a predicted classes
        # In our case it's 0 for negative tweet and 1 for positive.
        pc=model.predict_classes(t)
        pc=pc[0][0]
        # We can also can get the probablity of prediction been in a given class.
        # By default we get the probablity of being in class no. 1 which in our
        # case is probability of a tweet to be postive.
        # We can get the probablity of tweet being mean just by calculating 1-prob.
        prob=model.predict_proba(t)
        prob=prob[0][0]
        print('%s -%smean (%.2f%%)' % (tweet.rstrip(), (' ' if pc==0 else ' not '),(1-prob)*100))
    

# model = load_model('model.h5')