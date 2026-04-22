import pandas as pd

data = pd.read_csv("data/crop_data.csv")

big_data = pd.concat([data]*25, ignore_index=True)

big_data.to_csv("data/crop_data_large.csv", index=False)

print("Large dataset created!")