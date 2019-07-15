import numpy as np
import pandas as pd

df = pd.read_csv(
    'fertility.csv'
)
df = df.drop(['Season'], axis=1)
# print(df)
# print(df.columns.values.tolist())

features = [
    'Age', 'Childish diseases', 'Accident or serious trauma',
    'Surgical intervention', 'High fevers in the last year',
    'Frequency of alcohol consumption', 'Smoking habit',
    'Number of hours spent sitting per day', 'Diagnosis']

num = ['Age', 'Number of hours spent sitting per day']

group_bool = ['Childish diseases', 'Accident or serious trauma', 'Surgical intervention']
df[group_bool] = df[group_bool].apply(lambda x: pd.Series(x).map({'yes' : 1, 'no': 0}))

dfnew = df.drop(['Diagnosis'], axis=1)
dfnew = pd.get_dummies(dfnew)


# ======================================= Machine Learning =====================================================
from sklearn.metrics import classification_report
X = dfnew
Y = df['Diagnosis']

# Split --------------------------------------------------------------
from sklearn.model_selection import train_test_split

xtrain , xtest, ytrain, ytest = train_test_split(
    X,
    Y,
    test_size = .1
    )

# ================================= Random Forest Classifier =================================
from sklearn.ensemble import RandomForestClassifier
model_etc = RandomForestClassifier(n_estimators= 50)
model_etc.fit(xtrain,ytrain)
prediksi_etc = model_etc.predict(xtest)
skor_etc = model_etc.score(xtest, ytest)
# print(prediksi_etc)
# print(skor_etc)
# ================================= LOGISTIC REGRESSION =================================
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(solver='liblinear')
model_log.fit(xtrain,ytrain)
prediksi_log = model_log.predict(xtest)
skor_log = model_log.score(xtest, ytest)
# print(model_log)
# print(skor_log)

# ================================= DECISION TREE CLASSIFIER =================================
from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier()
model_DT.fit(xtrain,ytrain)
prediksi_DT = model_DT.predict(xtest)
skor_DT = model_DT.score(xtest, ytest)
# print(model_DT)
# print(skor_DT)

# ['Age' 'Childish diseases' 'Accident or serious trauma'
#  'Surgical intervention' 'Number of hours spent sitting per day'
#  'High fevers in the last year_less than 3 months ago'
#  'High fevers in the last year_more than 3 months ago'
#  'High fevers in the last year_no'
#  'Frequency of alcohol consumption_every day'
#  'Frequency of alcohol consumption_hardly ever or never'
#  'Frequency of alcohol consumption_once a week'
#  'Frequency of alcohol consumption_several times a day'
#  'Frequency of alcohol consumption_several times a week'
#  'Smoking habit_daily' 'Smoking habit_never' 'Smoking habit_occasional']

print('Arin, prediksi kesuburan: {} (Random Forest Classifier)'.format(model_etc.predict([[29,	0,	0,	0,	5,	0,	0,	1,	1,	0,	0,	0,	0,	1,	0,	0]])[0]))
print('Arin, prediksi kesuburan: {} (Logistic Regression)'.format(model_log.predict([[29,	0,	0,	0,	5,	0,	0,	1,	1,	0,	0,	0,	0,	1,	0,	0]])[0]))
print('Arin, prediksi kesuburan: {} (Decision Tree Classifier)'.format(model_DT.predict([[29,	0,	0,	0,	5,	0,	0,	1,	1,	0,	0,	0,	0,	1,	0,	0]])[0]))
print()
print('Bebi, prediksi kesuburan: {} (Random Forest Classifier)'.format(model_etc.predict([[31,	0,	1,	1,	24,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0]])[0]))
print('Bebi, prediksi kesuburan: {} (Logistic Regression)'.format(model_log.predict([[31,	0,	1,	1,	24,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0]])[0]))
print('Bebi, prediksi kesuburan: {} (Decision Tree Classifier)'.format(model_DT.predict([[31,	0,	1,	1,	24,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0]])[0]))
print()
print('Caca, prediksi kesuburan: {} (Random Forest Classifier)'.format(model_etc.predict([[25,	1,	0,	0,	7,	1,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))
print('Caca, prediksi kesuburan: {} (Logistic Regression)'.format(model_log.predict([[25,	1,	0,	0,	7,	1,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))
print('Caca, prediksi kesuburan: {} (Decision Tree Classifier)'.format(model_DT.predict([[25,	1,	0,	0,	7,	1,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))
print()
print('Dini, prediksi kesuburan: {} (Random Forest Classifier)'.format(model_etc.predict([[28,	0,	1,	1,	24,	0,	0,	1,	0,	1,	0,	0,	0,	1,	0,	0]])[0]))
print('Dini, prediksi kesuburan: {} (Logistic Regression)'.format(model_log.predict([[28,	0,	1,	1,	24,	0,	0,	1,	0,	1,	0,	0,	0,	1,	0,	0]])[0]))
print('Dini, prediksi kesuburan: {} (Decision Tree Classifier)'.format(model_DT.predict([[28,	0,	1,	1,	24,	0,	0,	1,	0,	1,	0,	0,	0,	1,	0,	0]])[0]))
print()
print('Enno, prediksi kesuburan: {} (Random Forest Classifier)'.format(model_etc.predict([[42,	1,	0,	0,	8,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))
print('Enno, prediksi kesuburan: {} (Logistic Regression)'.format(model_log.predict([[42,	1,	0,	0,	8,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))
print('Enno, prediksi kesuburan: {} (Decision Tree Classifier)'.format(model_DT.predict([[42,	1,	0,	0,	8,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0]])[0]))