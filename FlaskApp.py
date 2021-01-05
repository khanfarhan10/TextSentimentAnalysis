# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pickle
import preprocessing

# load the model from disk
clf = pickle.load(open('nb_clf.pkl', 'rb'))
cv=pickle.load(open('tfidf_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if(len(message)>2):
            text = [message]
            data = preprocessing.text_Preprocessing(text)
            vect = cv.transform(data)
            my_prediction = clf.predict(vect)
        else:
            my_prediction=3
        
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
