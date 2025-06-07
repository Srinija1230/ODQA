# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re, math
import urllib
from googletrans import Translator
import urllib3
import warnings
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import inflect 
p = inflect.engine() 
import string
import os

def projection_simplex(v, z=1):
    """
    Projection onto the simplex:
        w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection onto the simplex.
        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize labels.
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Initialize primal and dual coefficients.
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))

        # Pre-compute norms.
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indices.
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]

                # All-zero samples can be safely ignored.
                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for the ith sample.
                delta = self._solve_subproblem(g, y, norms, i)

                # Update primal and dual coefficients.
                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            

            if vratio < self.tol:
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)
def que_classification():
    f=open("predefined.txt","r",encoding='utf-8')
    classes=[] #all the labels of the questions
    query=[]   #queries corresponding to the classes
    new_label=[] #labels in numerical representation
    for line in f:
        line=line.rstrip('\n')
        classes.append((line.split()[0]).split(":")[0])  #appending labels from the test to classes list
        query.append(line[5:])                           #appending queries from text to query list
    j=list(set(classes))   #set of the labels
    k=[i for i in range(len(j))] #numbers for labels
    my_dict={} 
    for i in range(7):
        my_dict.update({j[i]:k[i]}) #dictionary {label:number}
    for i in classes:
        new_label.append(my_dict[i]) #labels in numeric representation
    #query.append(given_query)
    #vectorising all the inputs
    vectorizer =TfidfVectorizer()
    #X = vectorizer.fit_transform(classes) #Learn vocabulary and idf, return document-term matrix (input-raw_document)
    Y = vectorizer.fit_transform(query)
    z=Y.toarray()
    #print(Y[-1])
    X = z[:len(new_label)]#y[:1342]
    y = new_label[:len(new_label)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1250)
    lr = MulticlassSVM(C=0.5, tol=0.001, max_iter=100, random_state=0, verbose=1)
    lr.fit(X_train,y_train)
    return (vectorizer,lr,my_dict)
    
def predict_ques(que):
    query=[]
    query.append(que)
    withbias=vectorizer.transform(query).toarray()
    maxx=0
    maxi=0
    for i in range(7):
        #score=(withbias*lr.weights[i])
        scorei=np.dot(withbias, lr.coef_[i].T)
        if(scorei >= maxx):
            maxx=scorei
            maxi=i
    for i in my_dict:
            if my_dict[i]==maxi:
                print("\nLabel of the given query :",i) 
                break
    return i
(vectorizer,lr,my_dict)=que_classification()

java_path = "C:/Program Files/Java/jdk1.8.0_151/bin/java.exe"
os.environ['JAVAHOME'] = java_path

warnings.filterwarnings('ignore')
prelink="https://www.bing.com/search?"
#from urllib.request import urlopen
def preprocess(x):
    # Step1 to lowercase
    lowercase=x.lower()
    temp_str = lowercase.split() 
    # Step2 removing digits
    new_string = [] 
    p = inflect.engine()
    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word) 
  
    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    # Step-3 removing punctuations
    translator = str.maketrans('', '', string.punctuation) 
    temp_str = temp_str.translate(translator) 
    # Step-4 removing stop words
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(temp_str) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    filtered_text =' '.join(filtered_text)
    # step-5 Lemmatise
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(filtered_text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas            
                
def NER_other_way(qc):
    print("qc : ",qc)
     
    nlp = spacy.load('en_core_web_sm')
    #stanford_ner_tagger = StanfordNERTagger('C:/Users/B151119/Desktop/SummerInternship/Week-5/stanford-ner-4.0.0/classifiers/english.muc.7class.distsim.crf.ser.gz','C:/Users/B151119/Desktop/SummerInternship/Week-5/stanford-ner-4.0.0/stanford-ner.jar')
    ners1 ={'PERSON':[],'ORG':[],'CARDINAL':[],'TIME':[],'DATE':[],'GPE':[],'PERCENT':[],'LAW':[],'ORDINAL'
        :[],'NORP':[],'LOC':[],'EVENT':[],'FAC':[],'QUANTITY':[],'MONEY':[]}
    for i in range(len(Top_similarity_sent)):
        sentence=Top_similarity_sent[i]
        doc = nlp(sentence)
        for ent in doc.ents: 
            if(ent.label_ in ners1.keys()):
                ners1[ent.label_].append(ent.text)
   ners=ners1
    ners['LOCA'] = ners.pop('GPE')
    ners['ORGA'] = ners.pop('ORG')
    ners['NUMB'] = ners.pop('CARDINAL')
    ners['PERC'] = ners.pop('PERCENT')
    ners['PERS'] = ners.pop('PERSON')
    ners['CUR'] = ners.pop('MONEY')
    ners['QUANT']=ners.pop('QUANTITY')
   
    lis = ners['LOC']
    for i in lis:
        ners['LOCA'].append(i)

    return (ners1,ners[qc])

def qquery(qu):
    # BING
    print("BING SEARCH")
    address = "http://www.bing.com/search?q=%s" % (urllib.parse.quote_plus(str(qu)))
    http = urllib3.PoolManager()
    resp = http.request('GET',address)
    soup = BeautifulSoup(resp.data)
    print("Top 10 URLS:\n----------------------------------------------------------------\n\n")
    if not(soup.findAll('li', attrs={'class':'b_algo'})):
        print("Not able to find urls, Some Error came call back the function again....\n")
        qquery(qu)

    for listt in soup.findAll('li', attrs={'class':'b_algo'}):
        head_links = listt.find('h2')
        for link in head_links.find_all('a', href=True):
            Top_10_urls.append(link['href'])
            print(link['href']+"\n")
            Whole_output.append(link['href']+"\n")
        
    return Top_10_urls


def translate(to_translate, to_langage="auto", langage="auto"):
    try:
        translator = Translator()
        result = translator.translate(to_translate, src='te', dest='en')
        quest.append(result.text)
        Whole_output.append(result.text)        
        return result
    except:
        print("Translate error")
        return translate(to_translate, to_langage="auto", langage="auto")

def Sentence_Extraction():
    Numlinks=len(Top_10_urls)
    #sen=open("Whole_sentence.txt","w")
    i=0
    for i in range(Numlinks):
        try:
            nP=0
            http = urllib3.PoolManager()
            resp = http.request('GET',Top_10_urls[i])
            #content=urllib2.urlopen(Top_10_urls[i]) ############ May be Here you may get ERROR
            htmlSource=resp.data	############ May be Here you may get ERROR
            contentt = BeautifulSoup(htmlSource)
            
            for para in contentt.findAll(['p']):
                pa=para.text
                whole_sent.append(pa.strip()) #all sentence writing into one list line by line
                #sen.write(pa.encode('UTF-8')+"\n") # all sentence writing into one file line by line
                nP=nP+1 # number of imp content paragraphs in each URL...........
        except IOError as e:
            print(" IO ERROR "+e)	   		
        except ValueError:
            print("VALUE ERROR ")
        except:
            print("ERROR ")
            Top_10_urls.pop(i)
            Sentence_Extraction()
            break

    

def Finding_Similarity_Matrix(q):
    cc=0
    for i in range(len(whole_sent)):
        stmt=whole_sent[i].split(".")
        for j in range(len(stmt)):
            if(len(stmt[j].split("?"))==1):
                cc=cc+1
                all_sen.append(stmt[j])
    WORD = re.compile(r'\w+')
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return(0.0)
        else:
            return(float(numerator) / denominator)

    def text_to_vector(text):
        words = WORD.findall(text)
        return(Counter(words))
    text1 =q  # given user query
    x=0
    
    for n in range(len(all_sen)):
        text2=all_sen[n]
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)

        cosine = get_cosine(vector1, vector2)
        x=x+1
        consine_list.append(cosine)



def Most_Relative_Sentence():
	
	num=10 ##number of most relative sentences considering
	if(len(whole_sent)<num):
		num = len(whole_sent)
	tt= (sorted(consine_list, reverse=True)[:num])
	for k in range(num):
		l=[i for i,j in enumerate(consine_list) if j==tt[k]]
		myString = all_sen[l[0]].strip()
		sent = ' '.join(myString.split())
		Top_similarity_sent.append(sent)
		Whole_output.append(sent+"\n")
	return Top_similarity_sent

def que_ner(qc_label,sent_list):
    ners1 ={'PERSON':[],'ORG':[],'CARDINAL':[],'TIME':[],'DATE':[],'GPE':[],'PERCENT':[],'LAW':[],'ORDINAL'
        :[],'NORP':[],'LOC':[],'EVENT':[],'FAC':[],'QUANTITY':[],'MONEY':[]}
    for i in range(len(sent_list)):
        sentence=sent_list[i]
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
        for ent in doc.ents: 
            if(ent.label_ in ners1.keys()):
                ners1[ent.label_].append(ent.text)
    ners=ners1
    ners['LOCA'] = ners.pop('GPE')
    ners['ORGA'] = ners.pop('ORG')
    ners['NUMB'] = ners.pop('CARDINAL')
    ners['PERC'] = ners.pop('PERCENT')
    ners['PERS'] = ners.pop('PERSON')
    ners['CUR'] = ners.pop('MONEY')
    ners['QUANT']=ners.pop('QUANTITY')
    lis = ners['LOC']
    for i in lis:
        ners['LOCA'].append(i)
    for i in range(len(ners[qc_label])):
        ners[qc_label][i]=ners[qc_label][i].lower()
    result=ners[qc_label]
    return (ners1,result)

############################################# Calling QA System Step by Step #########################################################
import pandas as pd
import time
input_file = open("updated_file_847_que","r",encoding='utf-8').readlines()
Questions = []
actual_ans = []
final_ans = []
labels=[]
#####FILE OPENED IN WRITE MODE#####
def findingmostsimilar(poss , similar):
    besind=100
    best=[]
    print(poss)
    for j in poss:
        for i in range(len(similar)):
            #tok = similar[i].split(" ")
            if(j in similar[i]):
                if(besind>=i):
                    besind=i
                    best.append(j)
    return best[0]
def ners(ner):
    possibles =[]
    for i in ner.keys():
        for j in ner[i]:
            possibles.append(j)
    return possibles
def find_max_freq(que_ner,ner,Top_similarity_sent):
    possibles = ners(ner)
    que_ners = ners(que_ner)
    for i in ner.keys():
        for j in ner[i]:
            possibles.append(j)
    words, counts = np.unique(possibles, return_counts=True)
    corpus_df = pd.DataFrame({'WORD':words, 'COUNT':counts})
    corpus_df.sort_values('COUNT', ascending=False, inplace=True)
    corpus_df.reset_index(drop=True,inplace=True)
    find=corpus_df[corpus_df.columns[0]]
    samefreq=[]
    maxfreq=0
    for i in range(len(find)):
        if(find[i].lower() not in que_ners):
            freq=int(corpus_df[corpus_df['WORD']==find[i]]['COUNT'])
            if(maxfreq<=freq):
                samefreq.append(find[i])
                maxfreq=freq
            else:
                break

    if(len(samefreq)==1):
        return samefreq[0]
    elif(len(samefreq)>1):
        return findingmostsimilar(samefreq,Top_similarity_sent)
    return " SORRY NO ANSWER"

def cleanup_funct():
	Top_10_urls=[]
	whole_sent=[]
	consine_list=[]
	all_sen=[]
	Top_similarity_sent=[]
	ans=[]
	Sentence_NER=[]
	ner=[]
	quest=[]
	q_ner=[]
	q_text=[]
	Whole_output=[]
	final_answers=[]
	ss=[]
	label_ans=[]
	ans_type=[]
	all_sen1=[]
	random_ans=[]
	return (Top_10_urls,whole_sent,consine_list,all_sen,Top_similarity_sent,ans,
			Sentence_NER,ner,quest,q_ner,q_text,Whole_output,
			final_answers,label_ans,ss,ans_type,all_sen1, random_ans)
for line in input_file:
    #print(line)
    line=line.strip()
    line=line.split("@@@")
    labels.append(line[0])
    actual_ans.append(line[1].split(":::")[0])
    Questions.append(line[1].split(":::")[1])
start=time.time()
for l in range(len(Questions)): #running from 0 to 186
    qa=Questions[l]
    print("\n---------------------------------------------------------\n\n")
    print(" QUESTION ")
    print(l)
    #for qa in Questions:
    (Top_10_urls,whole_sent,consine_list,all_sen,Top_similarity_sent,ans,Sentence_NER,ner,quest,q_ner,q_text,Whole_output,final_answers,
    label_ans,ss,ans_type,all_sen1,random_ans ) = cleanup_funct()
    print("\n---------------------------------------------------------\n\n")
    q=translate(qa,'en','te') # translate te-->eng
    print("\n--------------------------------------------------------\n\n")
    print("Telugu Sentence:\t"+qa+"\n")
    print("English Sentence:\t"+q.text+"\n")
    print("\n\n")
    #print("Top 10 URLS:\n----------------------------------------------------------------\n\n")
    Top_10_urls = qquery(q.text) # calling get Top 10 URLS
    print("---------------------------------------------------------\n\n")
    Sentence_Extraction()  # calling to get whole content from 10 urls(well define sentences only)
    Finding_Similarity_Matrix(q.text) # calling find similarity matri between query and all sentences
    Top_similarity_sent = Most_Relative_Sentence() # calling to print higest top 5 similarity senteces.
    print("\n---------------------------------------------------------\n\n")
    qc=predict_ques(qa) # user query question classification
    print('qc :',qc)
    (ner,possibles) = NER_other_way(qc)
    print("\n------------------------------------------------------------\n\n")
    print("possible answers:")
    print(possibles)
    words, counts = np.unique(possibles, return_counts=True)
    corpus_df = pd.DataFrame({'WORD':words, 'COUNT':counts})
    corpus_df.sort_values('COUNT', ascending=False, inplace=True)
    corpus_df.reset_index(drop=True,inplace=True)
    find=corpus_df[corpus_df.columns[0]]
    (ner1,lower) = que_ner(qc, [q.text])
    found=0
    if(len(find)==0):
        ans=find_max_freq(ner1,ner,Top_similarity_sent)
        print("FINAL ANSWER\n")
        print(ans)
        final_ans.append(ans)
    else:
        
        samefreq=[]
        maxfreq=0
        #print(find)
        #print(corpus_df)
        for i in range(len(find)):
            if(find[i].lower() not in lower):
                found=1
                #print(find[i])
                #print(corpus_df['WORD'])
                freq=int(corpus_df[corpus_df['WORD']==find[i]]['COUNT'])
                if(maxfreq<=freq):
                    samefreq.append(find[i])
                    maxfreq=freq
                else:
                    break

        if(len(samefreq)==1):
            print("FINAL ANSWER:\n")
            print(samefreq[0])
            final_ans.append(samefreq[0])
        elif(len(samefreq)>1):
            ans=findingmostsimilar(samefreq,Top_similarity_sent)
            print("FINAL ANSWER:\n")
            print(ans)
            final_ans.append(ans)

        if(found==0):
            ans=find_max_freq(ner1,ner,Top_similarity_sent)
            print("FINAL ANSWER:\n")
            print(ans)
            final_ans.append(ans)

end=time.time()
print("\n-----------------------------------------------------------\n\n")
print("ACTUAL ANS\n")
print(actual_ans)
print("\n-----------------------------------------------------------\n\n")
print("FINAL ANS\n")
print(final_ans)
print("TOTAL TIME FOR QUESTIONS in sec :")

print(end-start)
# FINDIND ACCURACY NOW FOR ALL THE TEST DATA (SAMPLE QUESTIONS)

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
def time_format(actual_ans):
    for i in range(len(actual_ans)):
        x=actual_ans[i]
        x=x.lower()
        x=x.strip()
        #print(x)
        
        #print(x)
        
        hhmmss='(0?[1-9]|1[0-2]|[0-9]*) ?(hours|hour|hrs) ?(0?[0-5][0-9]|[0-9]*) ?(minutes|minute|min) ?(0?[0-5][0-9]|[0-9]*) ?(seconds|second|sec)'   #12 hours 30 min 20 sec
        thhmmss=re.search(hhmmss,x)

        hms='(0?[1-9]|1[0-2]|[0-9]*) ?: ?(0?[0-5][0-9]|[0-9]*) ?: ?(0?[0-5][0-9]|[0-9]*) ?(hours|hour|hrs|hr)'   #12:30:20 hours
        thms=re.search(hms,x)

        hhmm='(0?[1-9]|1[0-2]|[0-9]*) ?(hours|hour|hrs|hr) ?(0?[0-5][0-9]|[0-9]*) ?(minutes|minute|min)'      #12 hours 30 min
        thhmm=re.search(hhmm,x)

        hm='(0?[1-9]|1[0-2]|[0-9]*) ?: ?(0?[0-5][0-9]|[0-9]*) ?(hours|hour|hrs|hr)'  #12:30 hours
        thm=re.search(hm,x)

        hs='(0?[1-9]|1[0-2]|[0-9]*) ?(hours|hour|hrs|hr) ?(0?[0-5][0-9]|[0-9]*) ?(seconds|second|sec)'   #12 hours 30 sec
        ths=re.search(hs,x)

        #hr='(0?[1-9]|1[0-2]|[0-9]*) [\s]?(hours|hour|hrs|hr)'   #12 hours
        hr='(0?\d+)[\s]?(hours|hour|hrs|hr)' 
        thr=re.search(hr,x) 

        mmss='(0?[0-5][0-9]|[0-9]*) ?(minutes|minute|min) ?(0?[0-5][0-9]|[0-9]*) ?(seconds|second|sec)'  #30 min 20 sec
        tmmss=re.search(mmss,x)

        ms='(0?[0-5][0-9]|[0-9]*) ?: ?(0?[0-5][0-9]|[0-9]*) ?(minutes|minute|min)'   #30:20 minutes
        tms=re.search(ms,x)

        #min='(0?[0-5][0-9]|[0-9]*) [\s]?(minutes|minute|min)'  #30 min
        min='(0?\d+)[\s]?(minutes|minute|min)'
        tmin=re.search(min,x)

        #sec='(0?[0-5][0-9]|[0-9]*) [\s]?(seconds|second|sec)' #30 sec
        sec='(0?\d+)[\s]?(seconds|second|sec)' 
        tsec=re.search(sec,x)

        #xx=x.strip()
        #print(x)

        hour1='(one hour|one hr|an hour|an hr|a hour|a hr|hour|hr)'
        thour1=re.search(hour1,x) 

        min1='(one minute|one min|a minute|a min|minute|min)'
        tmin1=re.search(min1,x)

        sec1='(one second|one sec|a second|a sec|second|sec)'
        tsec1=re.search(sec1,x)
        found=0
        total_sec=0
        if thhmmss:
           #print(thhmmss.group())
           hours_to_sec=int(thhmmss.group(1))*(3600)
           min_to_sec=int(thhmmss.group(3))*(60)
           total_sec=int(thhmmss.group(5))+hours_to_sec+min_to_sec
           #print(total_sec)
           found=1
        elif thms:
           #print(thms.group())
           hours_to_sec=int(thms.group(1))*(3600)
           min_to_sec=int(thms.group(2))*(60)
           total_sec=int(thms.group(3))+hours_to_sec+min_to_sec
           #print(total_sec)
           found=1
        elif thhmm:
           #print(thhmm.group())
           hours_to_sec=int(thhmm.group(1))*(3600)
           min_to_sec=int(thhmm.group(3))*(60)
           total_sec=hours_to_sec+min_to_sec
           #print(total_sec)
           found=1
        elif thm:
           #print(thm.group())
           hours_to_sec=int(thm.group(1))*(3600)
           min_to_sec=int(thm.group(2))*(60)
           total_sec=hours_to_sec+min_to_sec
           #print(total_sec)
           found=1
        elif ths:
           #print(ths.group())
           hours_to_sec=int(ths.group(1))*(3600)
           total_sec=hours_to_sec+int(ths.group(3))
           #print(total_sec)
           found=1
        elif thr:
           #print(thr.group())
           total_sec=int(thr.group(1))*(3600)
           #print(total_sec)
           found=1
        elif tmmss:
           #print(tmmss.group())
           min_to_sec=int(tmmss.group(1))*(60)
           total_sec=int(tmmss.group(3))+min_to_sec
           #print(total_sec)
           found=1
        elif tms:
           #print(tms.group())
           min_to_sec=int(tms.group(1))*(60)
           total_sec=int(tms.group(2))+min_to_sec
           #print(total_sec)
           found=1
        elif tmin:
           #print(tmin.group())
           total_sec=int(tmin.group(1))*(60)
           #print(total_sec)
           found=1
        elif tsec:
           #print(tsec.group())
           total_sec=int(tsec.group(1))
           #print(total_sec)
           found=1
        elif thour1:      
            #print(thour1.group())
            total_sec=3600
            #print(total_sec)
            found=1
        elif tmin1:
            #print(tmin1.group())
            total_sec=60
            #print(total_sec) 
            found=1
        elif tsec1:      
            #print(tsec1.group())
            total_sec=1
            #print(total_sec)
            found=1
        if(found==1):
            actual_ans[i]=str(total_sec)
    return actual_ans
        
    
def preprocess(actual_ans):
    rem=['.',',','?','!',')','(','\n']
    for i in range(len(actual_ans)):
        actual_ans[i]=actual_ans[i].lower()
        actual_ans[i]=" ".join(actual_ans[i].split())
        element = actual_ans[i].split(" ")
        lis=[]
        for j in element:
            sublist=[]
            for ii in j:
                if ii in rem:
                    sublist.append("")
                else:
                    sublist.append(ii)
            lis.append("".join(k for k in sublist))
        actual_ans[i]=" ".join(k for k in lis)
        stop_words = set(stopwords.words("english")) 
        word_tokens = word_tokenize(actual_ans[i]) 
        filtered_text = [word for word in word_tokens if word not in stop_words]
        actual_ans[i]=" ".join(ele for ele in filtered_text)
    return actual_ans
import math
def date_format(actual_ans):
    for i in range(len(actual_ans)):
        mon=['january','february','march','april','may','june','july','august','september','october','november','december']
        rem=['th','st','rd','nd']
        tok=actual_ans[i].split(" ")
        ans=[]
        for j in mon:
            if j in tok:
                ans.append(j)
                found=0
                for k in tok:
                    if k not in mon:
                        if(k.isdigit() and math.floor(math.log(int(k), 10)+1)==4):
                            last=k
                            found=1
                        elif(k.isdigit() and math.floor(math.log(int(k), 10)+1)==2):
                            ans.append(k)
                        else:
                            for ii in rem:
                                if(len(k.split(ii))>1):
                                    ans.append(k.split(ii)[0])
                                    find=1
                                    break
                if(found==1):
                    ans.append(last)
                actual_ans[i]=" ".join(ele for ele in ans)
                break
                
    return actual_ans
            
            
            
def rem(pred):
    lis=["times","percent","%","approximately","place","nearly","days"]
    p=pred.split(" ")
    s=[]
    for j in p:
        if j.lower() not in lis:
            s.append(j.lower())
    return(" ".join(ele for ele in s))
    
def formatt(actual_ans,pred_ans):
    actual_ans=preprocess(actual_ans)
    pred_ans=preprocess(pred_ans)
    from word2number import w2n
    for i in range(len(actual_ans)):
        if(actual_ans[i].isdigit()):
            try:  
                # initializing string 
                test_str = pred_ans[i]

                # printing original string 
                #print("The original string is : " + test_str) 

                # Convert numeric words to numbers 
                # Using word2number 
                res = w2n.word_to_num(test_str) 
                pred_ans[i]=str(res)
                # printing result  
                #print("The string after performing replace : " + str(res))

            except ValueError:
                pass
        else:
            pred_ans[i]=rem(pred_ans[i])
            actual_ans[i]=rem(actual_ans[i])
    actual_ans=date_format(actual_ans)
    pred_ans=date_format(pred_ans)
    actual_ans=time_format(actual_ans)
    pred_ans=time_format(pred_ans)
    return (actual_ans,pred_ans)

def findingacc(actual_ans,pred_ans,file3):

    (actual_ans,pred_ans)=formatt(actual_ans,pred_ans)
    from difflib import SequenceMatcher
    exact_match=0
    partial_match=0
    file_label = open(file3,'r',encoding='utf-8').readlines()
    
    
    for i in range(len(actual_ans)):
        #932 range
        
        if (actual_ans[i]==pred_ans[i]):
            exact_match+=1
        else:
            #partial match 0.7 threshold
            x=SequenceMatcher(None,actual_ans[i],pred_ans[i]).ratio()
            if (x>=0.7):
                partial_match+=1
            else:
                if(file_label[i].split("@@@")[0]=="PERS"):
                    tokens=pred_ans[i].split(" ")
                    for k in tokens:
                        if(k in actual_ans[i]):
                            partial_match+=1
                            break
                
    print("\n EXACT MATCH : \n"+str(exact_match*100/len(pred_ans))+"%")
    print("\n PARTIAL MATCH: \n"+str((partial_match+exact_match)*100/len(pred_ans))+"%")
    
findingacc(actual_ans,final_ans,"updated_file_847_que")




