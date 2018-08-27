import pandas as pd
import re

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df = pd.read_table('reviews.csv', names = ['0'], dtype = str)
df = pd.DataFrame(data=df)
df = df.drop(df.index[0])
df = pd.DataFrame(df['0'].str.split('|',1).tolist(), columns = ['labels','reviews'])
myStopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}



def tokenizeLabels(x):
    if x == 'positive':
        return 1
    else:
        return 0

df['labels'] = df['labels'].apply(lambda x: tokenizeLabels(x))

df['reviews'] = df['reviews'].apply(lambda x : str.lower(x))
df['reviews'] = df['reviews'].str.replace(r"[\',]",'')
df['reviews'] = df['reviews'].apply(lambda x : " ".join(re.findall('[\w]+',x)))


def filterWords(inputSentence):
    wordTokens = inputSentence.split(' ')
    filteredSentence = []
    for w in wordTokens:
        if w.isalpha() and w not in myStopWords:
            filteredSentence.append(w)
    return filteredSentence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

review_train, review_test, label_train, label_test = \
train_test_split(df['reviews'], df['labels'], test_size=0.2)

from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


mlpClassifier = MLPClassifier(hidden_layer_sizes=(5,),verbose=True,learning_rate_init=0.05,max_iter=200)
pipeline = Pipeline([('bow',CountVectorizer(analyzer = filterWords,min_df=2,ngram_range=(1,2))),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',mlpClassifier)])
pipeline.fit(review_train,label_train)

predictions = pipeline.predict(review_test)
print (classification_report(predictions,label_test))
print (confusion_matrix(label_test,predictions))
print (pd.crosstab(label_test, pipeline.predict(review_test), rownames=['True'], colnames=['Predicted'], margins=True))

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

auc_score = metrics.roc_auc_score(label_test, pipeline.predict_proba(review_test)[:,1])
fpr_svc, tpr_svc, _ = roc_curve(label_test, pipeline.predict_proba(review_test)[:,1])
roc_auc = metrics.auc(fpr_svc,tpr_svc)
plt.title('ROC Curves')
plt.plot(fpr_svc, tpr_svc, color='blue', label='NN ROC curve (area = %0.2f)' % roc_auc)


default_prob = pipeline.predict_proba(review_test)[:,1]
confusion_mat = confusion_matrix(label_test, pipeline.predict(review_test))
results = classification_report(label_test, pipeline.predict(review_test))

decision_tree_model = DecisionTreeClassifier(max_depth=25)
pipeline = Pipeline([('bow',CountVectorizer(analyzer = filterWords, stop_words=myStopWords,min_df=2,ngram_range=(1,2))),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',decision_tree_model)])
pipeline.fit(review_train,label_train)

predictions = pipeline.predict(review_test)
print (classification_report(predictions,label_test))
print (confusion_matrix(label_test,predictions))


fpr_svc, tpr_svc, _ = roc_curve(label_test, pipeline.predict_proba(review_test)[:,1])
roc_auc = metrics.auc(fpr_svc,tpr_svc)
plt.plot(fpr_svc, tpr_svc, color='orange', label='DT ROC curve (area = %0.2f)' % roc_auc)


default_prob = pipeline.predict_proba(review_test)[:,1]
confusion_mat = confusion_matrix(label_test, pipeline.predict(review_test))
results = classification_report(label_test, pipeline.predict(review_test))


myVectorizer = CountVectorizer(analyzer = filterWords, stop_words=myStopWords,min_df=2,ngram_range=(1,2))
pipeline = Pipeline([('bow',myVectorizer),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])
pipeline.fit(review_train,label_train)
predictions = pipeline.predict(review_test)