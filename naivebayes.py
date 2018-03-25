import pandas as pd
import numpy as np
import re,os,glob

# get dataset
file_train = os.getcwd()+'/training_tokenized_no_punc.csv'
file_test = os.getcwd()+'/testing_tokenized_no_punc.csv'

data = pd.read_csv(file)
data_testing = pd.read_csv(file_test)
data['description'].fillna('',inplace=True)
data_testing['description'].fillna('',inplace=True)


######################### naivebayes #########################
# use pipeline to do naivebayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])

train_result = text_clf.fit(data['description'], data['Priori Sub-Category'])
predicted = text_clf.predict(data_testing['description'])

# preformance of naive_bayes
from sklearn import metrics

test_names = list(data_testing['Priori Sub-Category'].unique())
report_nb = metrics.classification_report(data_testing['Priori Sub-Category'], predicted,target_names = test_names)
#print(report_nb)
precision,recall,f1_score,support = re.findall('total(.*?)\n',report_nb)[0].split()
print('as a summary, we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision,recall,f1_score))

######################### SVM #########################
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),])

train_result_svm = text_clf_svm.fit(data['description'], data['Priori Sub-Category'])
predicted_svm = text_clf_svm.predict(data_testing['description'])

# preformance of SVM
report_svm = metrics.classification_report(data_testing['Priori Sub-Category'], predicted_svm,target_names = test_names)
precision_2,recall_2,f1_score_2,support = re.findall('total(.*?)\n',report_svm)[0].split()
#print(report_svm)
print('as a summary, we have \n\n precision:{0}, \n\n recall:{1}, \n\n f1_score:{2}'.format(precision_2,recall_2,f1_score_2))
