# BASA-Competition--Text-Classification

The aim of this repository is to do the text classification on apple store app description, then decide the app's category.

There are two raw datasets and four code file:

1.testing.csv (sample test dataset)
2.training.csv (sample training dataset)
3.data_cleaning.py (remove special characters(like *), punctuations, stopwords, stemming of training data and test data)
4.naibayes_SVM.py (try naive_bayes and svm as baseline)
5.other models.py (try other models like MultinomialNB, SGDClassifier, RandomForestClassifier, AdaBoostClassifier using global fit)
6.out of core.py (compare with global fit, use partial fit as learning method to save time and internal memory, and try models like
  MultinomialNB, SGDClassifier, passive aggressive classifier)


Result: The passive aggressive classifier performs best based on global fit, F1: 0.84, recall: 0.84, precision: 0.84


Data Description:
Here is four columns in dataset, "id","description","main category","sub category".
id: app id
description: text description about app
main category: main category the app belong to
sub category: detailed category the app belong to


Order: run data_cleaning.py -> naibayes_SVM.py -> other models.py -> out of core.py
(the code about MultinomialNB, SGDClassifier models in "other models.py ","out of core.py "(global fit) are same)

