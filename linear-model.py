import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

customers = pd.read_csv('Ecommerce-Customers.txt')
sns.pairplot(customers)
plt.title('Correlations')
plt.show()


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
plt.title('Linear regression')
plt.show()
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

plt.scatter(y_test, predictions)
plt.title('Model Testing')
plt.show()
sns.distplot((y_test-predictions), bins=50)
plt.title('Distribution for Model Testing')
plt.show()

modelcf = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficients'])
print(modelcf)
