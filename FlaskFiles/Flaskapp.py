# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pickle
import preprocessing
from textblob import TextBlob
from gensim.summarization import keywords
from gensim.summarization import summarize
import spacy

# load the model from disk
clf = pickle.load(open('nb_clf.pkl', 'rb'))
cv=pickle.load(open('tfidf_model.pkl','rb'))
app = Flask(__name__)

# load the spacy english model
nlp = spacy.load("en_core_web_sm")

#home
@app.route('/')
def home():
	return render_template('home.html')

#Sentiment
@app.route('/nlpsentiment')
def sentiment_nlp():
    return render_template('sentiment.html')

@app.route('/sentiment',methods = ['POST','GET'])
def sentiment():
    if request.method == 'POST':
        message = request.form['message']
        text = [message]
        data = preprocessing.text_Preprocessing(text)
        vect = cv.transform(text)
        my_prediction = clf.predict(vect)
        Keywords = keywords(message)
        Keywords = 'Keywords =' + str(Keywords)
        new = Keywords.split('\n')
        newstr = ''
        for word in new:
            newstr = newstr + word + ', '
        sum_message = summarize(message)
        overview = TextBlob(message)
        Polarity = round(overview.sentiment.polarity,2)
        Polarity = 'Polarity =' + str(Polarity)
        Subjectivity = round(overview.sentiment.subjectivity,2)
        Subjectivity = 'Subjectivity =' + str(Subjectivity)
        return render_template('sentiment.html', prediction=my_prediction,newstr=newstr,sum_message=sum_message,Polarity=Polarity,Subjectivity=Subjectivity)


if __name__ == '__main__':
	app.run(debug=True)
