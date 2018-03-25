import pandas as pd
import numpy as np
import re,os,glob,time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix, hstack
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

def get_minibatch(stream,size,index):
    docs = list(stream['description'].iloc[size*index:size*(index+1)])
    y = list(stream['Priori Sub-Category'].iloc[size*index:size*(index+1)])
    return docs,y

def get_dummybatch(dummy,size,index):
    batch = dummy[size*index:size*(index+1)]
    return batch

#customize your file path
path = os.getcwd()+'/'
train = pd.read_csv(path+'training_tokenized_no_punc.csv')
test = pd.read_csv(path+'testing_tokenized_no_punc.csv')
train['description'].fillna('',inplace=True)
test['description'].fillna('',inplace=True)
classes = list(test['Priori Sub-Category'].unique())

#if want to test bigram, change ngram_range=(1,1) to ngram_range=(2,2)
# use HashingVectorizer instead of CountVectorizer
start = time.time()
count_vect = HashingVectorizer(decode_error='ignore',
                               n_features=2 ** 18,
                               alternate_sign=False,
                               ngram_range=(1,1),
                               analyzer='word',
                               norm = 'l2',
                               token_pattern = '([a-zA-Z]+)')

vect_transformer =  count_vect.fit(train['description'])
train_counts = vect_transformer.transform(train['description'])
tfidf_transformer = TfidfTransformer().fit(train_counts)
tfidf = tfidf_transformer.transform(train_counts)
end = time.time()
elapsed = end-start
print('HashingVectorizer time elapsed: {}'.format(elapsed))

test_counts = vect_transformer.transform(test['description'])
tfidf_test = tfidf_transformer.transform(test_counts)

################################ fit without main Category ################################
clf = MultinomialNB()
#clf = SGDClassifier(loss='log',random_state=1,max_iter=1)
#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
#clf = PassiveAggressiveClassifier(random_state=42)

# train model
#train models based on out_of_core learning method (partial fit)
l = len(train)
num = 5
for i in range(num):
    start = time.time()
    x_train,y_train = get_minibatch(train,size=int(l/num),index=i)
    x_train = count_vect.transform(x_train)
    clf.partial_fit(x_train,y_train,classes = classes)
    end = time.time()
    elapsed = end - start
    print('out of core train time(without main category) iter {}, time elapsed:{}'.format(i,elapsed))
# prediction
start = time.time()
predicted = clf.predict(tfidf_test)
end = time.time()
elapsed = end-start
print('out of core prediction time(without main category) time elapsed: {}'.format(elapsed))
# performance
report = metrics.classification_report(test['Priori Sub-Category'], predicted,target_names = classes)
precision,recall,f1_score,support = re.findall('total(.*?)\n',report)[0].split()
print('out of core prediction performance(without main category), we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision,recall,f1_score))

################################ fit with main Category ################################
train_dummy = np.array(pd.get_dummies(train['Priori Category']))
test_dummy = np.array(pd.get_dummies(test['Priori Category']))
#train_dummy = np.array(pd.get_dummies(train['Priori Category']))*10
#test_dummy = np.array(pd.get_dummies(test['Priori Category']))*10
train_combined = hstack([tfidf, train_dummy])
test_combined = hstack([tfidf_test, test_dummy])

#clf_2 = MultinomialNB()
#clf_2 = SGDClassifier(loss='log',random_state=1,max_iter=1)
clf_2 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
#clf_2 = PassiveAggressiveClassifier(random_state=42)
#clf_2 = PassiveAggressiveClassifier(random_state=0)

#train models based on out_of_core learning method (partial fit)
l = len(train)
num = 5
for i in range(num):
    start = time.time()
    x_train_2,y_train_2 = get_minibatch(train,size=int(l/num),index=i)
    dummy = get_dummybatch(train_dummy,size=int(l/num),index=i)
    x_train_2 = count_vect.transform(x_train_2)
    train_combined = hstack([x_train_2, dummy])
    clf_2.partial_fit(train_combined,y_train_2,classes = classes)
    end = time.time()
    elapsed = end - start
    print('out of core train time(with main category) iter {}, time elapsed:{}'.format(i,elapsed))

# prediction
start = time.time()
predicted_2 = clf_2.predict(test_combined)
end = time.time()
elapsed = end-start
print('out of core prediction time(with main category) time elapsed: {}'.format(elapsed))
# performance
report_2 = metrics.classification_report(test['Priori Sub-Category'], predicted_2,target_names = classes)
precision,recall,f1_score,support = re.findall('total(.*?)\n',report_2)[0].split()
print('out of core prediction performance(without main category), we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision,recall,f1_score))

################################ use global data( not out of core method) ################################
train_dummy = np.array(pd.get_dummies(train['Priori Category']))
test_dummy = np.array(pd.get_dummies(test['Priori Category']))
train_combined = hstack([tfidf, train_dummy])
test_combined = hstack([tfidf_test, test_dummy])

#clf_3 = MultinomialNB()
#clf_3 = SGDClassifier(loss='log',random_state=1,max_iter=1)
clf_3 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
#clf_3 = PassiveAggressiveClassifier(random_state=42)
#clf_3 = PassiveAggressiveClassifier(random_state=0)

start = time.time()
train_result_combined = clf_3.fit(train_combined, train['Priori Sub-Category'])
end = time.time()
elapsed = end-start
print('global train time elapsed: {}'.format(elapsed))

start = time.time()
predicted_combined = clf_3.predict(test_combined)
end = time.time()
elapsed = end-start
print('global prediction time elapsed: {}'.format(elapsed))

report_combined = metrics.classification_report(test['Priori Sub-Category'], predicted_combined,target_names = classes)
precision,recall,f1_score,support = re.findall('total(.*?)\n',report_combined)[0].split()
print('global data performance, we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision,recall,f1_score))
