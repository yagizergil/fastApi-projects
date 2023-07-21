import pandas as pd
import numpy as np

df1 = pd.read_excel('LabeledText.xlsx')
df1.columns = ["File Name", "TEXT_IN_ENGLISH","label"]
df1.drop('File Name', inplace=True, axis=1)

df2 = pd.read_csv('all-data6.csv')
# df3 = pd.concat([df2 , df1], axis=0 , ignore_index=True)

print(df1)
# df3.to_csv("all-data6.csv")