#Import neccessary modules 
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#Store the OnionOrNot csv file, containing news headlines from both the Onion and real life. 0 = real life 1 = Onion
df = pd.read_csv('OnionOrNot.csv')

#Function that inputs text and 'cleans' it by lowering it and removing punctuation 
def clean_text(text):
  text = text.strip().lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

#Store non-cleaned test for comparing human and machine accuracy 
old_text = np.array(df['text'])

#Apply the clean_text function to all news headlines in df
df['text'] = df['text'].apply(lambda text: clean_text(text))
#Store X(news headlines) and y(whether its real or fake) into numpy arrays, where there are an equal amount of fake and real articles
X_fake = np.array(df['text'][df['label']==0])[:9000]
X_real = np.array(df['text'][df['label']==1])[:9000]
X = np.concatenate((X_fake, X_real))
y = np.array([0]*9000+[1]*9000)

#The training variables will consist of 70% of the dataset, and the testing variables will consist of 30%
X_train, y_train = X[:16800], y[:16800]
X_test, y_test = X[16800:], y[16800:]

#Create the model using sklearn's Pipeline, which pipelines transformers with a final estimator for predictions
model = Pipeline([
    ('count', CountVectorizer()), #Convert the X headlines to a matrix of word tokens and their frequencies within each headline
    ('tfidf', TfidfTransformer()), #Normalize the count matrix while scaling down the impact of very frequent word tokens which provide little information
    ('scale', MaxAbsScaler()), #Scale each feature in the TFIDF matrix by its maximum absolute value to increase linearity
    ('log', LogisticRegression()) #Each vector in the matrix is now a data point for the Logistic Regression algorithm 
])
#Fit the data on X_train and y_train
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #Test accuracy is ~84.41%

pickle.dump(model, open('model.pkl', 'wb')) #Save the model
