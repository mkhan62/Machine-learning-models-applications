import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline


# feature exploration
yelp = pd.read_csv('yelp.csv')
yelp['text_length'] = yelp['text'].apply(len)
g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, x=yelp['text_length'], bins=50)
plt.show()

sns.boxplot(x='stars', y='text_length', data=yelp)
plt.show()

sns.countplot('stars', data=yelp)
plt.show()

yelp_star = yelp.groupby('stars').mean()
yelp_star_corr = yelp_star.corr()
sns.heatmap(yelp_star_corr, cmap='plasma')
plt.show()

# dataset
X = yelp['text']
y = yelp['stars']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('ml', MultinomialNB())
])

print('hi again')
pipe.fit(X_train, y_train)
predict = pipe.predict(X_test)
print(classification_report(y_test, predict))

