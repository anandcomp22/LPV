from mpi4py import MPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = pd.read_csv("ml/data/crop_data_large.csv")

le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

chunk_size = len(data) // size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size if rank != size - 1 else len(data)

chunk = data.iloc[start_idx:end_idx]

X = chunk.drop("label", axis=1)
y = chunk["label"]

start_time = time.time()

model = RandomForestClassifier()
model.fit(X, y)

end_time = time.time()

y_pred = model.predict(X)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_pred)

time_taken = end_time - start_time
times = comm.gather(time_taken, root=0)
accuracies = comm.gather(acc, root=0)

print(f"Process {rank} | Rows: {len(chunk)} | Acc: {acc:.4f} | Time: {time_taken:.4f}s")

if rank == 0:
    print("\n=== FINAL RESULT ===")
    print(f"Processes used: {size}")
    print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Average Time: {sum(times)/len(times):.4f}s")