# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 01:11:02 2018

@author:  Olakunle Oladiran
"""
from __future__ import print_function
#from Abbrev import ABBREVIATIONS
from contractions import CONTRACTION_MAP
#from wordfreq import WordFrequency
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
#from EnglishLangFreq import EngFreq
import pandas.io.sql as pds
import pandas as pd
import numpy as np
import pyodbc
from sklearn.model_selection import train_test_split
import spacy
from nltk import FreqDist
from spacy.lang.en import English
import scipy.sparse as sp
from numpy.linalg import norm
from scipy import stats
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import os
from sklearn.decomposition import NMF
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
#from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# import gensim
# from gensim.models import word2vec
from sklearn.manifold import TSNE
#from pattern3.en import tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#import func
import importlib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
# import wordfreq
# from wordfreq import word_frequency
# from wordfreq import zipf_frequency
#importlib.reload(func)

def connect(query,params = None):
    conn_str = (
    r'DRIVER={ODBC Driver 13 for SQL Server};'
    r'SERVER=localhost\SQLEXPRESS;'
    r'DATABASE=mimiciii;'
    r'Trusted_Connection=yes;')
    cnxn = pyodbc.connect(conn_str)
    #query ='"""{}"""'.format(query)
    query =query 
    df = pds.read_sql(query, cnxn,params=params)
    return df
    
def Dataframe(corpus):
    dfID = [x for x in corpus['NoteID']]
    dfDocType = [x for x in corpus['CAT']]   
    dfNotes = [x for x in corpus['TEXT']]
   
    return dfID,dfDocType, dfNotes

def Transform(corpus,doc_tp='UNKNOWN'):
    DF = pd.DataFrame(corpus)
    DType = [doc_tp for x in range(len(DF))]
    vector =  pd.DataFrame({ 'NoteID':DF['NoteID'],'Doc_Type':DType,'TEXT': DF['TEXT']})
    return vector

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']
wnl = WordNetLemmatizer()



def remove_generic_text(note):
    clean = re.sub(
        r'\[\*\*[a-zA-Z0-9]{1,10}\/[a-zA-Z0-9]{1,10}\s\([0-9]{1,3}\)\s[0-9]{1,10}\**\]|\[\*\*\d{1,4}\-\d{1,4}\*\*\]|\[\*\*\d{4}\-\d{2}\-\d{2}\*\*\]*\s(\s|\s?\d{2}\:\d{2}\s[a-zA-Z]{1,2})|\[\*\*[a-zA-Z0-9]{1,10}\s(\*\*\]|[a-zA-Z]{1,5}\s[a-zA-Z]{1,10}\s[a-zA-Z0-9]{1,10}|\([A-Za-z0-9]{1,10}\)\s[a-zA-Z0-9]{1,10}\*\*\])|\[\*\*[a-zA-Z0-9]{1,10}\s[a-zA-Z0-9]{1,10}\s[a-zA-Z0-9]{1,10}\*\*\]|\*\*[a-zA-Z0-9]{1,10}\-[a-zA-Z0-9]{1,10}\-[A-Za-z0-9]{1,10}\*\*\]'," ",note)
    return clean

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        elif pos_tag.startswith('U'):
            return wn.NOUN
        else:
            return None
    
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

def lemmatize_text(text):
    
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def sp_lemmatize_text(text):
    spacy_lemma = English()
    tokens = spacy_lemma(text)
    lemmas = [token.lemma_ for token in tokens]    
    lemmatized_text = ' '.join(lemmas)
    return lemmatized_text


def replace_abbreviation(note):
    Doc = ''.join( ABBREVIATIONS.get( word, word) for word in re.split( '(\W+)', note ) )
    return Doc
#def replace_abbreviation(note):
    #xxx=[]
    #tokens = tokenize_text(note)
    #for line in text:
        #new_line = line
    for word in note.split():
        abb = ABBREVIATIONS.get(word)
        print(abb)
        #for f_key, f_value in ABBREVIATIONS.items():
#         if abb is not None:
#             newNote = note.replace(word,abb)
#                  #new_note = ' '.joint(newWord)
                   
#             return print(newNote[:2])


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


            
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z0-9]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus,tokenize=False):
    
    normalized_corpus = []    
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = remove_generic_text(text) 
        text = text.upper()
        #text = replace_abbreviation(text)
        text = text.lower()
#         if lemmatize:
#             text = sp_lemmatize_text(text)
#         else:
#             text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus  

def build_feature_matrix(documents, feature_type='tfidf', ngram_range=(1, 1), min_df=0.0, max_df=1.0,use_idf=False):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                     ngram_range=ngram_range,use_idf=use_idf)
    else:
        raise Exception("Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


# def compute_cosine_similarity(query_TFIDF, corpus_TFIDF, top_n=3):
#     similarity = cosine_similarity(query_TFIDF, corpus_TFIDF).flatten()
#     # get docs with highest similarity scores
#     top_docs = similarity.argsort()[::-1][:top_n]
#     top_docs_with_score = [(index, round(similarity[index], 3))for index in top_docs]
#     return top_docs_with_score


def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
   # compute similarities
    similarity = np.dot(doc_features, corpus_features.T)
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 2))for index in top_docs]
    return top_docs_with_score

def Tokenize_norm(note):
    toks = []
    for x in note:
        token = tokenize_text(x)
        toks.append(token)
    return toks
    

def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights,sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])
    topics = [np.vstack((terms.T, term_weights.T)).T
    for terms, term_weights in zip(sorted_terms, sorted_weights)]
    return topics

# print all the topics from a corpus
def print_topics_udf(topics, total_topics=1, weight_threshold=0.000000000, display_weights=False,num_terms=None):
    newtopic=[]
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic]
        topic = [(word, round(wt,8)) for word, wt in topic if abs(wt) >= weight_threshold]
        if display_weights==True:
            newtopic.append(topic)
            #print ('Topic #'+str(index+1)+' with weights')
            #print (topic[:num_terms] if num_terms else topic)
        elif display_weights==False:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print (tw[:num_terms] if num_terms else tw)
    return newtopic
    
def Text_only_norm(note):
    toks = []
    for x in note:
        x.lower()
        token = tokenize_text(x)
        filtered = [tok for tok in token if not re.findall ('[\d]+', tok)]
        filt_toks = ' '.join(filtered)
        toks.append(filt_toks)
    return toks

# def CreateDict(allToks, unique_toks):
    
#     a = FreqDist(allToks)
#     dic = dict.fromkeys(unique_toks, 0)
#     for x,y in dic.items():
#         for p,q in a.items():
#             if x==p:
#                 dic[x]+=q
#     return dic

def CreateDict(allToks):
    new = FreqDist(allToks)
    return new
    
#Create Unique Tokens
def Create_uniqueTokens(list_Tokens):
    unique = []
    flat_list = [item for sublist in list_Tokens for item in sublist]
    for i in flat_list:
        if not i in unique:
            unique.append(i)
    return flat_list, unique

# def CF(Toklist,unique_tokens):
#     CFDic = dict.fromkeys(unique_tokens, 0)
#     for x, y in enumerate(Toklist):
#         new = FreqDist(y)
#         for j,k in new.items():
#             for p,q in CFDic.items():
#                 if j==p:
#                      CFDic[p]+=1
#     return CFDic

def CF(Toklist,unique_tokens):
    CFDic = dict.fromkeys(unique_tokens, 0)
    for x, y in enumerate(Toklist):
        new = FreqDist(y)
        for p,q in CFDic.items():
                if new[p]>=1:
                    CFDic[p]+=1
    return CFDic


#Method for CHisquare
def bow_extractor(corpus, ngram_range=(0,1)):
    
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
    

def tfidf_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

from numpy import array 
def make_class(corpus):
    cls = []
    for x,y in enumerate(corpus):
        a = "{}".format(x+1)
        cls.append(a)
    c = array(cls)
    return c
# def feature_reduce(Notes):
#     alltoks=[]
#     5pct = 0.05*len(Notes)
#     for x in Notes:
#         toks = tokenize_text(x)
#         alltoks.append(toks)
#     token=[x for x in]

def Compute_RR(SummedDictTr, SummedDictTs,z_value,ToksTr,ToksTs):
    import math
    Feature_list =[]
    RRCILB=[]
    RRCIUB =[]
    RRplain = []
    RRlog = []
    Total_Tr =len(ToksTr)
    Total_Test =len(ToksTs)
    fivepct = 0.05*Total_Tr 
    for wordA, a in SummedDictTr.items():
        for wordB, c in SummedDictTs.items():
            if wordA == wordB and a!=0 and c!=0 and a>=fivepct and c>=fivepct:
                d = Total_Test - c
                b = Total_Tr - a 
                top = np.array([a,b])
                btm= np.array([c,d])
                stacked = np.vstack((top, btm))
                OR,pv = stats.fisher_exact(stacked) 
                R =  ((a*(Total_Test))/((c+0.00001)*(Total_Tr)))
                #R = a*(Total_Test)/(c*(Total_Tr)+.0001)
                sterr=math.sqrt(((1/a)+(1/(c+.0001)))-((1/Total_Tr)-(1/Total_Test)))
                lb = math.exp( math.log(R)-float(z_value)*sterr )
                ub = math.exp( math.log(R)+float(z_value)*sterr )
                #CISig = ub-lb
                #if R>=1.0000001 and wordA not in Feature_list:# and not re.findall('[\d]+',wordA):
                if wordA not in Feature_list and pv<=0.049 and lb>=1:# and not re.findall('[\d]+',wordA):    
                    Feature_list.append(wordA)
                    RRCILB.append(lb)
                    RRCIUB.append(ub)
                    RRplain.append(R)
                    #RRlog.append(R)
                   
    return Feature_list,RRCILB,RRCIUB,RRplain

def NMF_Tuples(Notes):
    NMFSuperList=[]
    NMFList=[]
    wds = []
    val = []
    vectorizer, tfidf_matrix = build_feature_matrix(Notes,feature_type='tfidf') 
    total_topics = 1 #len(norm_corpusTrain)
    nmf = NMF(n_components=total_topics, random_state=42, alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf_matrix)      
    feature_names = vectorizer.get_feature_names()
    weights = nmf.components_
    topics = get_topics_terms_weights(weights, feature_names)
    topicTuple = print_topics_udf(topics=topics, total_topics=total_topics,num_terms=None, display_weights=True)
    tuples = sorted(topicTuple[0],key=lambda x:x[1],reverse=False) 
    for x,y in tuples:  
        if y>0.06:
            NMFList.append(x) 
    return NMFList

def NMF_corpus(label,docList):
    newCorpus=[]
    for doc in docList:
        filtered = [word for word in label if word in doc]
        filtered_doc = ' '.join(filtered)
        newCorpus.append(filtered_doc)
    return newCorpus


def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()


import warnings
warnings.filterwarnings('ignore') 


