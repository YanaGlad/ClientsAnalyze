import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error

train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test.csv')

print("Посмотрим на случайные строки в датасете")
print(train_data.sample())
print(train_data.sample())
print(train_data.sample())
print("Посмотрим первые 5 строк датасета")
print(train_data.head())
print("Посмотрим последние 5 строк датасета")
print(train_data.tail())

for d in [train_data, test_data]:
    d['TotalSpent'] = d['TotalSpent'].replace(' ', 0)
    d['TotalSpent'] = d['TotalSpent'].astype(float)

num_cols_with_churn = ['ClientPeriod', 'MonthlySpending', 'TotalSpent', 'Churn']

#Числовые признаки
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

# Категориальные признаки
cat_cols = [
    'Sex',  # 0
    'IsSeniorCitizen',  # 1
    'HasPartner',  # 2
    'HasChild',  # 3
    'HasPhoneService',  # 4
    'HasMultiplePhoneNumbers',  # 5
    'HasInternetService',  # 6
    'HasOnlineSecurityService',  # 7
    'HasOnlineBackup',  # 8
    'HasDeviceProtection',  # 9
    'HasTechSupportAccess',  # 10
    'HasOnlineTV',  # 11
    'HasMovieSubscription',  # 12
    'HasContractPhone',  # 13
    'IsBillingPaperless',  # 14
    'PaymentMethod'  # 15
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'

train_data.info()
print('Нет пустых значений')

for i in range(len(cat_cols)):
    print(train_data[cat_cols[i]].value_counts())

train_data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)

fig, ax = plt.subplots(figsize=(14, 4))
sns.boxplot( x = train_data.ClientPeriod, y=train_data.Churn, orient='h')

fig, ax = plt.subplots(figsize=(14, 4))
sns.boxplot(x = train_data.MonthlySpending, y = train_data.Churn, orient='h')

train_data[num_cols_with_churn].hist(num_cols, figsize=(10, 5), bins=30)

print(train_data[target_col].value_counts())
print("Классы не являются сбалансированными т.к. значений класса 0 больше, чем значений класса 1")

label_encoder = LabelEncoder()
for col in cat_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col]  = label_encoder.transform(test_data[col])
    test_data[col]  = test_data[col].astype('category')
    train_data[col] = train_data[col].astype('category')

print(train_data)

X = train_data.drop(columns=['Churn'])
y = train_data['Churn']

dummy = pd.get_dummies(X[cat_cols])
X = pd.concat([X[num_cols], dummy], axis=1)

dummy = pd.get_dummies(test_data[cat_cols])
test_encoded = pd.concat([test_data[num_cols], dummy], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression()

model.fit(X_train, y_train)
test_pred = model.predict(X_test)

print('MSE:', mean_squared_error(y_test, test_pred))
print('F1 score:', f1_score(y_test, test_pred))
print('ROC-AUC:', roc_auc_score(y_test, test_pred))

params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
          "penalty": ["l1", "l2", "elasticnet"],
          "solver": ["lbfgs", "saga"],
          "random_state": [42]}

model = LogisticRegression()
model_cv = GridSearchCV(model, param_grid=params, cv=7)
model_cv.fit(X_train, y_train)


print("Best parameters: ", model_cv.best_params_)
print("Best score: ", model_cv.best_score_)

test_pred = model_cv.predict(X_test)

print('MSE:', mean_squared_error(y_test, test_pred))
print('F1 score:', f1_score(y_test, test_pred))
print('ROC-AUC:', roc_auc_score(y_test, test_pred))

import numpy as np

from catboost import CatBoostClassifier

X = train_data.drop(columns=['Churn'])
y = train_data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = { 'iterations':[100]
          ,'l2_leaf_reg':[1,3,5,10]
          ,'learning_rate':[0.001,0.01]
          ,'depth':[3,5,8,10]
          ,'loss_function':['CrossEntropy']
         }

model = CatBoostClassifier(cat_features = cat_cols, random_seed=42)
cat_cv = GridSearchCV(model, param_grid = params, cv=3)
cat_cv.fit(X_train, y_train)

test_pred = cat_cv.predict(X_test)
print('MSE:', mean_squared_error(y_test, test_pred))
print('F1 score:', f1_score(y_test, test_pred))
print('ROC-AUC:', roc_auc_score(y_test, test_pred))

cat_cv.best_estimator_.fit(X,y)
prediction = cat_cv.best_estimator_.predict_proba(test_data)[:, 1]
print('Prediction', prediction)

predicted_data = pd.DataFrame(prediction, index = np.arange(len(prediction)), columns=['Churn'])
predicted_data.to_csv('submissionN.csv', index_label='Id')