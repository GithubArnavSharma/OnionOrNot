#Import neccessary modules
import numpy as np
import pandas as pd
import string
import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Load the OnionOrNot csv file and make the two classes balanced
df = pd.read_csv('OnionOrNot.csv')
df = df.sort_values('label')[6000:]

def clean_text(text):
  text = text.lower().strip()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

#Clean up the headlines from the text column
df['text'] = df['text'].apply(lambda text: clean_text(text))

X_ = np.array(df['text'])
y_ = np.array(df['label'])

#Vectorize the input headlines
total = ' '.join(X_).split()
word2int = {word: index+1 for index, word in enumerate(np.unique(total))}
X = np.array([[word2int[word] for word in x.split()] for x in X_])

#Pad all the vectors in X and preform scaling
max_len = max([len(x) for x in X])
X = pad_sequences(X, max_len)
X = MinMaxScaler().fit_transform(X)

#Calculate the distances between each headline vector upon each other in order to remove inputs too closely related for Logistic Regression
distances = pairwise_distances(X)
#Store the locations of the distances that are 0 and less than 0.2 
distance_zero = np.where(distances == 0)
distance_little = np.where(distances < 0.2)

#Put tuple (row, column) of every distance less than 0.2 in list little
little = [(distance_little[0][i], distance_little[1][i]) for i in range(len(distance_little[0]))]
#Put tuple (row, column) of every instance equal to 0 in list little_zero
little_zero = [(distance_zero[0][i], distance_little[1][i]) for i in range(len(distance_zero[0]))]
#Put tuple (row, column) that is greater than zero but less than 0.2 in list little 
little = np.ravel([l for l in little if l[0] != l[1] and l not in little_zero])

#Store the text headlines and labels of the inputs that are not correlated into X and y 
X = np.array([X_[i] for i in range(len(X)) if i not in little])
y = np.array([y_[i] for i in range(len(y_)) if i not in little])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#Make the model and fit it into the training data 
model = Pipeline([
    ('count', CountVectorizer()), #Convert text documents to a matrix of token counts 
    ('tfidf', TfidfTransformer()), #Weight tokens to scale down the impact of common tokens 
    ('scale', MaxAbsScaler()), #Scale the tfidf matrix for Logistic Regression
    ('log', LogisticRegression()) #Use the Logistic Regression algorith to preform model fitting
])
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #85.2% test accuracy

pickle.dump(model, open('model.pkl', 'wb')) 
