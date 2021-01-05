import pyodbc
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk
import datetime
from nltk.corpus import stopwords
import json
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import wordpunct_tokenize
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=softwareproject;Trusted_Connection=yes')

'''
cursor.execute ("SELECT   [id],[title],[author],[text] ,[label] FROM [ethiccorna].[dbo].[train]")
data = cursor.fetchall()


'''
cursor = conn.cursor()
sql_calculate = ("INSERT INTO [softwareproject].[dbo].[sstub](bugtype,projectName,bugLineNum,bugNodeStartChar,bugNodeLength,fixLineNum,fixNodeStartChar,fixNodeLength,sourceBeforeFix,sourceAfterFix,sourceBeforeFixtoken,sourceAfterFixtoken)VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")

with open('sstubs.json', encoding="utf8") as f:
    json_from_file = json.load(f)


encoding = 'utf-8'


for i in  range(0,len(json_from_file)):
    wrong=str(word_tokenize(json_from_file[i]['sourceBeforeFix']))
    wrong=wrong.split(',')
    filterwrong=""
    filtercorrect=""
    for w in wrong:
        filterwrong=filterwrong+w+" "
    correct = word_tokenize(json_from_file[i]['sourceAfterFix'])
    for wr in correct:
        filtercorrect=filtercorrect+wr+" "

    try:
        cursor.execute(sql_calculate, (
            json_from_file[i]['bugType'], json_from_file[i]['projectName'], json_from_file[i]['bugLineNum'],
            json_from_file[i]['bugNodeStartChar'],
            json_from_file[i]['bugNodeLength'], json_from_file[i]['fixLineNum'], json_from_file[i]['fixNodeStartChar'],
            json_from_file[i]['fixNodeLength'], json_from_file[i]['sourceBeforeFix']
            , json_from_file[i]['sourceAfterFix'], filterwrong, filtercorrect))
        cursor.commit()
        print(datetime.datetime.now())
        print(i)

    except:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=softwareproject;Trusted_Connection=yes')







print(counter)
print(len(counter))
print(len(data)-len(counter))
for j in len(nofeature):
    print(nofeature[j])


