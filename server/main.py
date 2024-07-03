import sys
import os
import joblib

from DataLoader.dataload import load_dataset

import DataPreprocessing.datapreprocessing as dp
from DataPreprocessing.datapreprocessing import DataCleaning, LemmaTokenizer

from Evaluation.evaluation import precision_score_plot, confusion_matrix_plot

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


data_path = os.getcwd() + r'\data'
data = load_dataset(data_path + r'\IMDB-Dataset.csv')

train, test = train_test_split(data, test_size=.3, random_state=42, shuffle=True)
x_train = train['Reviews']
y_train = train['Label']
x_test = test['Reviews']
y_test = test['Label']

classifier = Pipeline(
    steps = [
        ('clean', DataCleaning()),
        ('vect', TfidfVectorizer(analyzer='word', tokenizer=LemmaTokenizer(), ngram_range=(1, 3), min_df=10, max_features=10000)),
        ('classifier', LogisticRegression(
            penalty='l2',
            dual=False,
            tol=0.001,
            C=1,
            solver='lbfgs',
            max_iter=100,
            multi_class='auto',
            verbose=0,
            warm_start=False,
            n_jobs=None
        ))
    ]
)

classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)
y_score = classifier.predict_proba(x_test)[:, 1]

print("Precision Score on test dateset for Logistic Regression: %s" % precision_score(y_test, y_predict, average='micro'))
print("AUC Score on test dateset for Logistic Regression: %s" % roc_auc_score(y_test, y_score ,multi_class='ovo', average='macro'))

f1_score_train_1 = f1_score(y_test, y_predict, average="weighted")
print("F1 Score test dateset for Logistic Regression: %s" % f1_score_train_1)
confusion_matrix_plot(y_test, y_predict)


model_path = os.getcwd() + r'\models\model'
joblib.dump(classifier, model_path + r'\classifier.pkl', compress=True)