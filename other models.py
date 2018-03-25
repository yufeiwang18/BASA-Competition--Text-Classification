import pandas as pd
import numpy as np
import re,os,glob,time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix, hstack
from sklearn import metrics

#customize your file path
path = os.getcwd()+'/'

train = pd.read_csv(path+'training_tokenized_no_punc.csv')
test = pd.read_csv(path+'testing_tokenized_no_punc.csv')
train['description'].fillna('',inplace=True)
test['description'].fillna('',inplace=True)

classes = list(test['Priori Sub-Category'].unique())

#vectorize and tfidf
start = time.time()
count_vect = CountVectorizer(ngram_range=(2,2),min_df=1,analyzer='word',token_pattern = '([a-zA-Z]+)')
vect_transformer =  count_vect.fit(train['description'])
train_counts = vect_transformer.transform(train['description'])
tfidf_transformer = TfidfTransformer().fit(train_counts)
tfidf = tfidf_transformer.transform(train_counts)
end = time.time()
elapsed = end-start
print('vectorize and tfidf time elapsed: {}'.format(elapsed))
test_counts = vect_transformer.transform(test['description'])
tfidf_test = tfidf_transformer.transform(test_counts)

# other models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#define classifier
clf = MultinomialNB()
#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
##don't bother these two, really bad
#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=100000)
#clf = AdaBoostClassifier()

# train model
start = time.time()
train_result = clf.fit(tfidf, train['Priori Sub-Category'])
end = time.time()
elapsed = end-start
print('global train(without main category) time elapsed: {}'.format(elapsed))

# test model
start = time.time()
predicted = clf.predict(tfidf_test)
end = time.time()
elapsed = end-start
print('global prediction(without main category) time elapsed: {}'.format(elapsed))

# performance
report = metrics.classification_report(test['Priori Sub-Category'], predicted,target_names = classes)
precision,recall,f1_score,support = re.findall('total(.*?)\n',report)[0].split()
print('global data performance(without main category), we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision,recall,f1_score))

# as in our dataset there is a main category that can be used as predictor
# to predict sub category, so we add it into our model
train_dummy = np.array(pd.get_dummies(train['Priori Category']))
test_dummy = np.array(pd.get_dummies(test['Priori Category']))
train_combined = hstack([tfidf, train_dummy])
test_combined = hstack([tfidf_test, test_dummy])

# train model
start = time.time()
train_result_combined = clf.fit(train_combined, train['Priori Sub-Category'])
end = time.time()
elapsed = end-start
print('global train(with main category) time elapsed: {}'.format(elapsed))

# test model
start = time.time()
predicted_combined = clf.predict(test_combined)
end = time.time()
elapsed = end-start
print('global prediction (with main category) time elapsed: {}'.format(elapsed))
# performance
report_combined = metrics.classification_report(test['Priori Sub-Category'], predicted_combined,target_names = classes)
precision_2,recall_2,f1_score_2,support_2 = re.findall('total(.*?)\n',report_combined)[0].split()
print('global data performance(with main category), we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision_2,recall_2,f1_score_2))
