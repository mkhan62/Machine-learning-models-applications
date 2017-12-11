import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


ad_data = pd.read_csv('advertising.csv')
sns.distplot(ad_data['Age'])
plt.title('Distribution of Age')
plt.show()

sns.pairplot(ad_data, hue='Clicked on Ad')
plt.show()

logmodel = LogisticRegression()
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

# test example entry for model
input1 = {'Daily Time Spent on Site': [2], 'Age': [1], 'Area Income': [0], 'Daily Internet Usage': [8]}
input2 = pd.DataFrame(input1)
output = logmodel.predict(input2)
print(output)
