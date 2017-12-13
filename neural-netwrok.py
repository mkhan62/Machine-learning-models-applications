import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sess = tf.Session()
notes = pd.read_csv('bank_note_data.csv')
print(notes.head())

# Exploratory Data Analysis

sns.pairplot(notes, hue='Class', palette='plasma')
plt.show()

# feature scaling
X = notes.drop('Class', axis=1).as_matrix()
y = notes['Class'].as_matrix()

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
model = learn.DNNClassifier(n_classes=2, hidden_units=[10, 20, 10],feature_columns=feature_columns)
model.fit(X_train, y_train, steps=200, batch_size=20)
predicts = model.predict(X_test)




