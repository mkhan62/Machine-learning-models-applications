import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# labelled data taken however labelled won't be used except for testing algorithm purposes
# Exploratory Data Analysis
data = pd.read_csv('College_Data.txt')
data.set_index('Unnamed: 0', inplace=True)
sns.lmplot('Room.Board', 'Grad.Rate', data=data, hue='Private', palette='coolwarm', fit_reg=False)
plt.xlabel('Room.Board')
plt.ylabel('Grad.Rate')
plt.show()

sns.lmplot('Outstate', 'F.Undergrad', data=data, hue='Private', palette='coolwarm', fit_reg=False)
plt.xlabel('Outstate')
plt.ylabel('F.Undergrad')
plt.show()

data[data['Private'] == 'Yes']['Outstate'].hist(bins=35, label='Yes')
data[data['Private'] == 'No']['Outstate'].hist(bins=35, label='No')
plt.legend()
plt.xlabel('Outstate')
plt.show()
data.loc['Cazenovia College', 'Grad.Rate'] = 100

data[data['Private'] == 'Yes']['Grad.Rate'].hist(bins=35, label='Yes')
data[data['Private'] == 'No']['Grad.Rate'].hist(bins=35, label='No')
plt.legend()
plt.xlabel('Grad.Rate')
plt.show()

data['Cluster'] = pd.get_dummies(data['Private']).drop('No', axis=1)
model = KMeans(n_clusters=2)
X = data.drop('Private', axis=1)
y = data['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train)
predict = model.predict(X_test)
print(classification_report(y_test, predict))

