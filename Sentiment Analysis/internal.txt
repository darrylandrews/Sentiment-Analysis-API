import nltk
from nltk import word_tokenize
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pickle
import re
import torch
import transformers
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader

import warnings
warnings.filterwarnings('ignore')

slang = pd.read_csv('Datasets/Combined_Slang_Dictionary.csv')
stopwords_id = pd.read_csv('Datasets/stopwords-id.txt',delimiter='\n')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

vectorizer = pickle.load(open('Pickles/TFIDF/TFIDF_vectorizer.sav', 'rb'))
tfidf_lr = pickle.load(open('Pickles/TFIDF/TFIDF_LR.sav', 'rb'))
tfidf_svc = pickle.load(open('Pickles/TFIDF/TFIDF_SVC.sav', 'rb'))

cbow_model = pickle.load(open('Pickles/Word2Vec/CBOW/cbow_model.sav', 'rb'))
cbow_lr = pickle.load(open('Pickles/Word2Vec/CBOW/cbow_LR.sav', 'rb'))
cbow_lr_ft = pickle.load(open('Pickles/Word2Vec/CBOW/cbow_LR_FT.sav', 'rb'))
cbow_svc = pickle.load(open('Pickles/Word2Vec/CBOW/cbow_SVC.sav', 'rb'))

skip_model = pickle.load(open('Pickles/Word2Vec/Skip/skip_model.sav', 'rb'))
skip_lr = pickle.load(open('Pickles/Word2Vec/Skip/skip_LR.sav', 'rb'))
skip_svc = pickle.load(open('Pickles/Word2Vec/Skip/skip_SVC.sav', 'rb'))

indobert_before = torch.load('Pickles/indoBERT/before.pth', map_location=torch.device('cpu'))
indobert_after = torch.load('Pickles/indoBERT/after.pth', map_location=torch.device('cpu'))

def data_preprocessing(review):
    temp = ""
    tokenized = word_tokenize(review.lower())
    
    for j, word in enumerate(tokenized):
        if word in stopwords_id['Stopwords'].values:
            continue
        else:
            temp = temp + word + " "
    
    if temp == "":
        temp = review
    
    temp = stemmer.stem(temp)
    return temp

def data_cleaning(review):
    review = re.sub(r'\d+',' ', review)
    review = re.sub(r'[^\w\s]',' ', review)
    review = re.sub(r'\s{2,}',' ', review)
    
    return data_preprocessing(review)

def slang_cleaning(review):
    temp = ""
    tokenized = word_tokenize(review.lower())                               
    
    for j, word in enumerate(tokenized):
        flag = word in slang['Before'].values                         
        
        if flag == True:
            slang_index = slang[slang['Before'] == word].index.values   
            word = slang.at[slang_index[0], 'After']
        
        temp = temp + word + " "          
        
    return data_cleaning(temp)  

def tfidf_vectorize(review):
    temp = []
    temp.append(review)
    
    tfidf = vectorizer.transform(temp)
    
    return tfidf_lr.predict(tfidf), tfidf_svc.predict(tfidf)

def word2vec(review):
    words = []

    tokenized = word_tokenize(review)

    for w in tokenized:
        words.append(w)                   
    
    cbow_vec = []
    skip_vec = []
    
    try:
        res_cbow = np.mean(cbow_model[words], axis=0)
        cbow_vec.append(res_cbow)
        
        res_skip = np.mean(skip_model[words], axis=0)
        skip_vec.append(res_skip)
    except KeyError:
        cbow_model.build_vocab([words], update=True)
        cbow_model.train(words, total_examples = -1, epochs=1)
        
        res_cbow = np.mean(cbow_model[words], axis=0)
        cbow_vec.append(res_cbow)
        
        skip_model.build_vocab([words], update=True)
        skip_model.train(words, total_examples = -1, epochs=1)
        
        res_skip = np.mean(skip_model[words], axis=0)
        skip_vec.append(res_skip)
    
    cbow_vec = np.array(cbow_vec)
    skip_vec = np.array(skip_vec)
    
    return cbow_lr.predict(cbow_vec), cbow_lr_ft.predict(cbow_vec), cbow_svc.predict(cbow_vec), skip_lr.predict(skip_vec), skip_svc.predict(skip_vec)

def indo_sentiment(sentence):
    subwords = tokenizer.encode(sentence)
    
    subwords_before = torch.LongTensor(subwords).view(1, -1).to(indobert_before.device)
    subwords_after = torch.LongTensor(subwords).view(1, -1).to(indobert_after.device)
    
    logits_b = indobert_before(subwords_before)[0]
    label_b = torch.topk(logits_b, k=1, dim=-1)[1].squeeze().item()
    
    logits_a = indobert_after(subwords_after)[0]
    label_a = torch.topk(logits_a, k=1, dim=-1)[1].squeeze().item()
    
    return logits_b, label_b, logits_a, label_a

def indo_bert(review):
    logits_bef, label_bef, logits_aft, label_aft = indo_sentiment(review)
    
    return i2w[label_bef], i2w[label_aft]