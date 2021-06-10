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

df = pd.read_csv('/content/drive/MyDrive/OnionOrNot.csv')
df = df.sort_values('label')[6000:]

def clean_text(text):
  text = text.lower().strip()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

df['text'] = df['text'].apply(lambda text: clean_text(text))

X_ = np.array(df['text'])
y_ = np.array(df['label'])

total = ' '.join(X_).split()
word2int = {word: index+1 for index, word in enumerate(np.unique(total))}

X = np.array([[word2int[word] for word in x.split()] for x in X_])
max_len = max([len(x) for x in X])
X = pad_sequences(X, max_len)

scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

distances = pairwise_distances(X)
distance_zero = np.where(distances == 0)
distance_little = np.where(distances < 0.2)

little = [(distance_little[0][i], distance_little[1][i]) for i in range(len(distance_little[0]))]
little_zero = [(distance_zero[0][i], distance_little[1][i]) for i in range(len(distance_zero[0]))]
little = np.ravel([l for l in little if l[0] != l[1] and l not in little_zero])

X = np.array([X_[i] for i in range(len(X)) if i not in little])
y = np.array([y_[i] for i in range(len(y_)) if i not in little])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('scale', MaxAbsScaler()),
    ('log', LogisticRegression())
])
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

pickle.dump(model, open('sample_data/model.pkl', 'wb'))
