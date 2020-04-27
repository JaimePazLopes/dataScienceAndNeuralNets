import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

df = pd.read_csv("..\\files\\kc_house_data.csv")

print(df.isnull().sum())
print(df.describe())
print(df.describe().transpose())

# sns.distplot(df["price"], bins=100)
# plt.show()

# sns.countplot(df["bedrooms"])
# plt.show()

print(df.corr()["price"].sort_values())

# plt.figure(figsize=(10,5))
# sns.scatterplot(x="price", y="sqft_living", data=df)
# plt.show()

# plt.figure(figsize=(10,6))
# sns.boxplot(x="bedrooms", y="price", data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x="price", y="lat", data=df)
# plt.show()

botton99 = df.sort_values("price", ascending=False).iloc[216:]

# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=botton99, edgecolor=None, alpha=0.2, palette="RdYlGn", hue="price")
# plt.show()

# sns.boxplot(x="waterfront", y="price", data=df)
# plt.show()

botton99 = botton99.drop("id", axis=1)
botton99["date"] = pd.to_datetime(df["date"])
botton99["year"] = botton99["date"].apply(lambda date: date.year)
botton99["month"] = botton99["date"].apply(lambda date: date.month)

# sns.boxplot(x="month", y="price", data=df)
# plt.show()

# df.groupby("month").mean()["price"].plot()
# plt.show()

# df.groupby("year").mean()["price"].plot()
# plt.show()

botton99 = botton99.drop("date", axis=1)
botton99 = botton99.drop("zipcode", axis=1)

X = botton99.drop("price", axis=1).values
y = botton99["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))

model.add(Dense(1,activation="relu"))

model.compile(optimizer="adam", loss="mse")

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400)

model.save("..\\files\\houseModel.h5")
model = load_model("..\\files\\houseModel.h5")

# history = pd.DataFrame(model.history.history)
# history.plot()
# plt.show()

predictions = model.predict(X_test)
print(mean_squared_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))

print(explained_variance_score(y_test, predictions))

# plt.figure(figsize=(12,8))
# plt.scatter(y_test, predictions)
# plt.plot(y_test, y_test, "r")
# plt.show()

singleHouse = botton99.drop("price", axis=1).iloc[0]

singleHouse = scaler.transform(singleHouse.values.reshape(-1,19))
prediction = model.predict(singleHouse)
print(prediction, botton99.head(1)["price"])




