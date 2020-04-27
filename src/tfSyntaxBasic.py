import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

df = pd.read_csv("..\\files\\fake_reg.csv")
print(df.head())

# sns.pairplot(df)
# plt.show()

X = df[["feature1", "feature2"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# model = Sequential([Dense(4, activation="relu"),
#                     Dense(2, activation="relu")
#                     Dense(1)])

model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="rmsprop", loss="mse")

model.fit(X_train, y_train, epochs=250)

# lossDF = pd.DataFrame(model.history.history)
# lossDF.plot()
# plt.show()

print(model.evaluate(X_test, y_test, verbose=0))

predictions = model.predict(X_test)

predictions = pd.Series(predictions.reshape(300,))
predictionsDF = pd.DataFrame(y_test, columns=["Test True Y"])
predictionsDF = pd.concat([predictionsDF, predictions], axis=1)
predictionsDF.columns = ["Test True Y", "Predictions"]
print(predictionsDF.head())

# sns.scatterplot(x="Test True Y", y="Predictions", data=predictionsDF)
# plt.show()

print(mean_absolute_error(predictionsDF["Test True Y"], predictionsDF["Predictions"]))
print(mean_squared_error(predictionsDF["Test True Y"], predictionsDF["Predictions"]))

new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
print(model.predict(new_gem))

model.save("..\\files\\myGemModel.h5")
loadedmodel = load_model("..\\files\\myGemModel.h5")
print(loadedmodel.predict(new_gem))







