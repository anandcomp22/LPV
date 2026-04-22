import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv("ml/data/crop_data_large.csv")

le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

X = data.drop("label", axis=1)
y = data["label"]

start = time.time()

model = RandomForestClassifier(n_jobs=-1)
model.fit(X, y)

end = time.time()

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print(f"Accuracy: {acc:.4f}")

print(f"Parallel Training Time: {end - start:.4f} seconds")