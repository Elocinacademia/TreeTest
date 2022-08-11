import pandas as pd


df = pd.read_csv("../../../data/data.csv")
print("Read data ....")

#Drop nun values

df.dropna(inplace=True)
df = pd.get_dummies(df)

#Split the dataset
sum_n = len(df)
cut_n = round(sum_n * 0.8)

df[:cut_n].to_csv("../../../data/train.csv",index=False)# 不保存行索引
df[cut_n:].to_csv("../../../data/test.csv",index=False)


import pdb; pdb.set_trace()
