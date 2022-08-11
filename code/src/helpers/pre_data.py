import pandas as pd


train = pd.read_csv("../../../data/data.csv")
print("Read data ....")

#Drop nun values
train.dropna(inplace=False)

