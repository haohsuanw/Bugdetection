
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyodbc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import math
from sklearn.preprocessing import normalize
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import model_selection, naive_bayes, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import fasttext
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
def readserver():
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=softwareproject;Trusted_Connection=yes')
    cursor = conn.cursor()
    cursor.execute("SELECT  [sourcecode],[label]FROM [softwareproject].[dbo].[Forbugtype]")
    return cursor.fetchall()

'''

def rescale(data):
    rescale=[]
    for i in range(len(data)):
        rescale.append([data[i].length,0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(rescale)
    for j in range(len(data)):
        data[j].length=round(rescaledX[j][0],6)

'''



'''def fast(data):
    Y=[]
    X=[]
    for j in range(len(data)):
        X.append(data[j].sourcecode)
        Y.append(data[j].label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

    file1 = open("train.txt", 'w', encoding='utf-8')



    for i in range(len(X_train)):
        l=Y_train[i]
        if l==0:
            labe='__label__0'
        else:
            labe='__label__1'

        sentenc=labe+" "+X_train[i]
        file1.write(sentenc+"\n")

    validate=[]
    for k in range(len(Y_test)):
        l = Y_test[k]
        if l == 0:
            labe = '__label__0'
        else:
            labe = '__label__1'
        validate.append([labe])

    model = fasttext.train_supervised(input='train.txt')
    model.save_model("train.bin")
    predict=model.predict(X_test)
    print("FastText Accuracy Score -> ", accuracy_score(predict[0], validate) * 100)
    print("FastText Precision Score -> ", precision_score(predict[0], validate,average='binary',pos_label='__label__1') * 100)
    print("FastText recall Score -> ", recall_score(predict[0], validate,average='binary',pos_label='__label__1') * 100)
    print("FastText F1 Score -> ", f1_score(predict[0], validate,average='binary',pos_label='__label__1') * 100)
'''

def fast(data):
    Y=[]
    X=[]
    for j in range(len(data)):
        X.append(data[j].sourcecode)
        Y.append(data[j].label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

    file1 = open("train.txt", 'w', encoding='utf-8')


















    for i in range(len(X_train)):
        l=Y_train[i]


        if l=='SWAP_ARGUMENTS':
            labe='__label__0'
        elif l=='OVERLOAD_METHOD_MORE_ARGS':
            labe='__label__1'
        elif l=='CHANGE_OPERATOR':
            labe='__label__2'
        elif l=='LESS_SPECIFIC_IF':
            labe='__label__3'
        elif l=='SWAP_BOOLEAN_LITERAL':
            labe='__label__4'
        elif l=='CHANGE_NUMERAL':
            labe='__label__5'
        elif l=='ODELETE_THROWS_EXCEPTION':
            labe='__label__6'
        elif l=='CHANGE_OPERAND':
            labe='__label__7'
        elif l=='CHANGE_IDENTIFIER':
            labe='__label__8'
        elif l=='ADD_THROWS_EXCEPTION':
            labe='__label__9'
        elif l == 'DIFFERENT_METHOD_SAME_ARGS':
            labe = '__label__10'
        elif l == 'CHANGE_UNARY_OPERATOR':
            labe = '__label__11'
        elif l == 'OVERLOAD_METHOD_DELETED_ARGS':
            labe = '__label__12'
        elif l == 'CHANGE_MODIFIER':
            labe = '__label__13'
        elif l == 'CHANGE_CALLER_IN_FUNCTION_CALL':
            labe = '__label__14'
        elif l == 'MORE_SPECIFIC_IF':
            labe = '__label__15'

        sentenc=labe+" "+X_train[i]
        file1.write(sentenc+"\n")

    validate=[]
    for k in range(len(Y_test)):
        l = Y_test[k]
        if l == 'SWAP_ARGUMENTS':
            labe = '__label__0'
        elif l == 'OVERLOAD_METHOD_MORE_ARGS':
            labe = '__label__1'
        elif l == 'CHANGE_OPERATOR':
            labe = '__label__2'
        elif l == 'LESS_SPECIFIC_IF':
            labe = '__label__3'
        elif l == 'SWAP_BOOLEAN_LITERAL':
            labe = '__label__4'
        elif l == 'CHANGE_NUMERAL':
            labe = '__label__5'
        elif l == 'ODELETE_THROWS_EXCEPTION':
            labe = '__label__6'
        elif l == 'CHANGE_OPERAND':
            labe = '__label__7'
        elif l == 'CHANGE_IDENTIFIER':
            labe = '__label__8'
        elif l == 'ADD_THROWS_EXCEPTION':
            labe = '__label__9'
        elif l == 'DIFFERENT_METHOD_SAME_ARGS':
            labe = '__label__10'
        elif l == 'CHANGE_UNARY_OPERATOR':
            labe = '__label__11'
        elif l == 'OVERLOAD_METHOD_DELETED_ARGS':
            labe = '__label__12'
        elif l == 'CHANGE_MODIFIER':
            labe = '__label__13'
        elif l == 'CHANGE_CALLER_IN_FUNCTION_CALL':
            labe = '__label__14'
        elif l == 'MORE_SPECIFIC_IF':
            labe = '__label__15'
        validate.append([labe])

    model = fasttext.train_supervised(input='train.txt')
    model.save_model("train.bin")
    predict=model.predict(X_test)
    print("FastText Accuracy Score -> ", accuracy_score(predict[0], validate) * 100)
    print("FastText Precision Score -> ", precision_score(predict[0], validate,average='macro') * 100)
    print("FastText recall Score -> ", recall_score(predict[0], validate,average='macro') * 100)
    print("FastText F1 Score -> ", f1_score(predict[0], validate,average='macro') * 100)




def lis(data):
    a,b,c,d=traintest(data)
    Trainls(a,b,c,d)


def traintest(data):
    dif=[]

    for i in range(len(data)):
        dif.append([data[i].sourcecode,data[i].label])

    df = pd.DataFrame(dif,columns=['sourcecode', 'label'])
    df.info()
    df1 = df[['sourcecode', 'label']]


    X=df.sourcecode
    Y=df.label
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)


    return X_train, X_test, Y_train, Y_test


def RNN():
    max_words = 1000
    max_len = 150
    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(279,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def Trainls(X_train, X_test, Y_train, Y_test):
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)

    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    model = RNN()
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    model.fit(sequences, Y_train, batch_size=128, epochs=10,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    accr = model.evaluate(test_sequences, Y_test)
    ypredict=model.predict(test_sequences_matrix)

    print("LSTM Precision Score -> ", precision_score(ypredict, Y_test,average='macro') * 100)
    print("LSTM recall Score -> ", recall_score(ypredict, Y_test,average='macro') * 100)
    print("LSTM F1 Score -> ", f1_score(ypredict, Y_test,average='macro') * 100)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


def tfidfngram(data):
    a,b=makemartix(data)
    return train(a,b)


def makemartix(data):
    text=[]
    featuresize=5000
    finalmatrix=[]

    label=[]
    for i in range(len(data)):
        text.append(data[i].sourcecode)
        finalmatrix.append([])
        label.append(data[i].label)
    Tfidf_vect = TfidfVectorizer(max_features=featuresize,ngram_range = (3,3))
    tfidf=Tfidf_vect.fit_transform(text)
    matrix=tfidf.toarray()
    for j in range(len(matrix)):
        newmatrix=matrix[j].tolist()
        finalmatrix[j]=finalmatrix[j]+newmatrix


    return finalmatrix,label


def train(X,Y):
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X, Y, test_size=0.3)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    Naive = GaussianNB()
    Naive.fit(Train_X, Train_Y)
    predictions_NB = Naive.predict(Test_X)
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
    print("Naive Bayes Precision Score -> ", precision_score(predictions_NB, Test_Y,average='macro') * 100)
    print("Naive Bayes recall Score -> ", recall_score(predictions_NB, Test_Y,average='macro') * 100)
    print("Naive Bayes F1 Score -> ", f1_score(predictions_NB, Test_Y,average='macro') * 100)

    SVM.fit(Train_X, Train_Y)
    SVM = svm.SVC(C=1.0,  gamma='auto')
    predictions_SVM = SVM.predict(Test_X)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
    print("SVM Precision Score -> ", precision_score(predictions_SVM, Test_Y,average='macro') * 100)
    print("SVM recall Score -> ", recall_score(predictions_SVM, Test_Y,average='macro') * 100)
    print("SVM F1 Score -> ", f1_score(predictions_SVM, Test_Y,average='macro') * 100)


'''
   finalmatrix[j]= np.c_[finalmatrix[j], matrix[j]]

        print(finalmatrix[j])
 for k in range(featuresize):
            .append(matrix[j][k])

'''




if __name__ == '__main__':
    data = readserver()
    lis(data)
