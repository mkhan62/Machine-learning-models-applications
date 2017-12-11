import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv('KNN_Project_Data.txt')

#sns.pairplot(data, hue='TARGET CLASS')
#plt.show()

scaler = StandardScaler()
scaler.fit(data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))
data_featured = pd.DataFrame(scaled_features, columns=data.columns[:-1])
X = data_featured
y = data['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

error_rate = []
for k in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error_rate.append(np.mean(predictions != y_test))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

model = KNeighborsClassifier(n_neighbors=30)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
