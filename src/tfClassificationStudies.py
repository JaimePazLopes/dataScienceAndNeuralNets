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

df = pd.read_csv("..\\files\\cancer_classification.csv")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(df.head())
print(df.info())
print(df.describe().transpose())

# sns.countplot(x="benign_0__mal_1", data=df)
# plt.show()

# df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
# plt.show()

# plt.figure(figsize=(12,12))
# sns.heatmap(df.corr())
# plt.show()

X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

earlyStop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)


time = datetime.now().strftime("%Y-%m-%d--%H%M")

logFolder = "..\\files\\log\\fit\\" + time

board = TensorBoard(log_dir=logFolder, histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch",
                    profile_batch=2, embeddings_freq=1)

model = Sequential()
model.add(Dense(30,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(15,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[earlyStop, board])

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))




