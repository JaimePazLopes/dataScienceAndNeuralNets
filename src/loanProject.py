import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_info = pd.read_csv('..\\files\\lending_club_info.csv',index_col='LoanStatNew')

print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

feat_info('mort_acc')

df = pd.read_csv('..\\files\\lending_club_loan_two.csv')

df.info()

# sns.countplot(x="loan_status", data=df)
# plt.show()

# plt.figure(figsize=(12,6))
# sns.distplot(df["loan_amnt"], kde=False)
# plt.show()

cor = df.corr().transpose()
print(cor)

# sns.heatmap(cor, annot=True)
# plt.show()

feat_info("installment")
feat_info("loan_amnt")

# sns.scatterplot(x="installment", y="loan_amnt", data=df)
# plt.show()

# sns.boxplot(x="loan_status", y="loan_amnt", data=df)
# plt.show()

print(df.groupby("loan_status")["loan_amnt"].describe())

print(df["grade"].unique())
print(df["sub_grade"].unique())

# sns.countplot(x="grade", hue="loan_status", data=df)
# plt.show()

kinds = df["sub_grade"].unique()
kinds.sort()

# plt.figure(figsize=(12,6))
# sns.countplot(x="sub_grade", data=df, order=kinds, hue="loan_status")
# plt.show()

subdata= df[df["sub_grade"]>"F0"]
kinds = subdata["sub_grade"].unique()
kinds.sort()

# plt.figure(figsize=(12,6))
# sns.countplot(x="sub_grade", data=subdata, order=kinds, hue="loan_status")
# plt.show()

df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})

print(df[["loan_repaid", "loan_status"]].head(30))

print(df.corr().transpose())

# df.corr().transpose()["loan_repaid"].drop("loan_repaid").sort_values().plot.bar()
# plt.show()

print(len(df))

print(len(df) - df.count()) # df.isnull().sum()

print(100 - df.count()/len(df) * 100)

feat_info("emp_title")
feat_info("emp_length")

print(df["emp_title"].nunique())

print(df["emp_title"].value_counts())

df.drop("emp_title", axis=1, inplace=True)
print(df["emp_length"].sort_values().value_counts())

subdata= df
kinds = subdata["emp_length"].dropna().unique()

def order(label):
    if "<" in label:
        return 0
    if "10" in label:
        return 10
    return int(label[0])

kinds = sorted(kinds, key=order)
print(kinds)
# plt.figure(figsize=(12,4))
# sns.countplot(x="emp_length", data=subdata, order=kinds, hue="loan_status")
# plt.show()

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/(emp_co+emp_fp)
# emp_len.plot(kind='bar')
# plt.show()

df.drop("emp_length", axis=1, inplace=True)

print(df.isnull().sum())

print(df[["title", "purpose"]])

df.drop("title", axis=1, inplace=True)

feat_info("mort_acc")

print(df["mort_acc"].value_counts())

print(df.corr()["mort_acc"].sort_values())

totalMean = df.groupby("total_acc").mean()["mort_acc"]

def fillMortACC(total, mort):
    if np.isnan(mort):
        return totalMean[total]
    return mort

df["mort_acc"] = df.apply(lambda x: fillMortACC(x["total_acc"], x["mort_acc"]), axis=1)

print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

print(df.select_dtypes(["object"]).columns)

def apply26or60(x):
    if "36" in x:
        return 36
    return 60

print(df["term"].value_counts())
df["term"] = df.apply(lambda x: apply26or60(x["term"]), axis=1)
print(df["term"].value_counts())

df.drop("grade", axis=1, inplace=True)

dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
df.drop("sub_grade", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)
df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
print(df.columns)

print(df["home_ownership"].value_counts())
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")

dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df.drop("home_ownership", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

df["zipcode"] = df["address"].apply(lambda address: address[-5:])
df.drop("address", axis=1, inplace=True)

dummies = pd.get_dummies(df["zipcode"], drop_first=True)
df.drop("zipcode", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

df.drop("issue_d", axis=1, inplace=True)

df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda address: int(address[-4:]))
df.drop("earliest_cr_line", axis=1, inplace=True)

df.drop("loan_status", axis=1, inplace=True)

X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(78, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(39, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(19, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))

losses = pd.DataFrame(model.history.history)

# losses.plot()
# plt.show()

predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

import random

random.seed(101)
random_int = random.randint(0, len(df))
newcustomer = df.drop("loan_repaid", axis=1).iloc[random_int]
newcustomer = scaler.transform(newcustomer.values.reshape(1,78))
predict = model.predict_classes(newcustomer)

print(predict, df.iloc[random_int]["loan_repaid"])









