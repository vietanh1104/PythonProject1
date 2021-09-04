from sklearn.datasets import fetch_20newsgroups

NewsClass = ['rec.sport.baseball', 'rec.sport.hockey']

DataTrain = fetch_20newsgroups(subset='train', categories=NewsClass, shuffle=True, random_state=42)
print(DataTrain.target_names)

print(len(DataTrain.data))
print(len(DataTrain.target))
#Extract features from text
from sklearn.feature_extraction.text import CountVectorizer
CountVect = CountVectorizer()
XTrainCounts = CountVect.fit_transform(DataTrain.data)
print(XTrainCounts.shape)
#Frequency of each word
from sklearn.feature_extraction.text import TfidfTransformer
TfTransformer = TfidfTransformer(use_idf=False).fit(XTrainCounts)
XTrainNew = TfTransformer.transform(XTrainCounts)
TfidfTransformer = TfidfTransformer()
XTrainNewidf = TfidfTransformer.fit_transform(XTrainCounts)
#build a classifier
from sklearn.naive_bayes import MultinomialNB
NBMutilclassifier = MultinomialNB().fit(XTrainNewidf,DataTrain.target)

NewsClassPred = NBMutilclassifier.predict(XTrainNewidf)

accuracy = 100 * (DataTrain.target== NewsClassPred).sum() / XTrainNewidf.shape[0]
print 'Accuracy = ',round(accuracy,2),'%'
