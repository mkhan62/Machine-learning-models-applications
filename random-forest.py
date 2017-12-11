import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Exploratory data Analysis
df = pd.read_csv('Loan_data.csv')
df[df['credit.policy'] == 1]['fico'].hist(bins=35, color='blue', label='Credit Policy 1')
df[df['credit.policy'] == 0]['fico'].hist(bins=35, color='red', label='Credit Policy 0')
plt.legend()
plt.xlabel('FICO')
plt.show()

df[df['not.fully.paid'] == 0]['fico'].hist(bins=35, color='blue', label='NFP 0')
df[df['not.fully.paid'] == 1]['fico'].hist(bins=35, color='red', label='NFP 1')
plt.legend()
plt.xlabel('FICO')
plt.show()

sns.countplot(df['purpose'], hue=df['not.fully.paid'])
plt.tight_layout()
plt.show()

sns.jointplot('fico', 'int.rate', df)
plt.show()

sns.lmplot('fico', 'int.rate', df, hue='not.fully.paid', col='not.fully.paid')
plt.show()

# coding categorical columns
purpose_list = ['debt_consolidation', 'credit_card', 'all_other', 'home_improvement', 'small_business', 'major_purchase', 'educational']
final_data = pd.get_dummies(df, columns=['purpose'], drop_first=True)

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# decision tree
dmodel = DecisionTreeClassifier()
dmodel.fit(X_train, y_train)
dec_pred = dmodel.predict(X_test)
print(classification_report(y_test, dec_pred))
print('\n')

# random forest
rmodel = RandomForestClassifier()
rmodel.fit(X_train, y_train)
ran_pred = rmodel.predict(X_test)
print(classification_report(y_test, ran_pred))

