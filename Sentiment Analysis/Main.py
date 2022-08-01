from internal import slang_cleaning, tfidf_vectorize, word2vec, indo_bert
from flask import Flask, request, render_template
import webbrowser
from threading import Timer
import sys
import os

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = str(request.form["reviews"])
    
    cleaned_review = slang_cleaning(review)
    
    tfidf_lr, tfidf_svc = tfidf_vectorize(cleaned_review)
    cbow_lr, cbow_lr_ft, cbow_svc, skip_lr, skip_svc = word2vec(cleaned_review)
    indo_before, indo_after = indo_bert(cleaned_review)
    
    return render_template('index.html', 
                           original_review = f'{review}', 
                           preprocess = f'{cleaned_review}',
                           res1=f'{tfidf_lr[0]}', 
                           res2=f'{tfidf_svc[0]}', 
                           res3=f'{cbow_lr[0]}', 
                           res4=f'{cbow_lr_ft[0]}', 
                           res5=f'{cbow_svc[0]}',
                           res6=f'{skip_lr[0]}',
                           res7=f'{skip_svc[0]}',
                           res8=f'{indo_before.capitalize()}', 
                           res9=f'{indo_after.capitalize()}')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run()