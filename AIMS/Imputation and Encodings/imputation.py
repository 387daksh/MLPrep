import pandas as pd
import numpy as np

dataset = {"age": [12, np.nan, 15, 17, 11, 24, np.nan, 45, np.nan, 44, 26, 18, np.nan]}
df = pd.DataFrame(dataset)
print(df)

non_empty = [x for x in df["age"] if pd.isnull(x) == False]

print("choose the type of imputation you would like")
print("1-mean, 2-median, 3-mode")
choice = int(input())

sorted_list = sorted(non_empty)
n = len(sorted_list)
if n % 2 == 1:
    median=sorted_list[n // 2]
else:
    median=(sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2

mean = int(sum(non_empty)/len(non_empty))

mode = pd.Series(non_empty).mode()[0]

if choice == 1:
    value = mean
    print("using mean for imputation:", mean)
elif choice == 2:
    value = median
    print("using median for imputation:", median)
elif choice == 3:
    value = mode
    print("using mode for imputation:", mode)
else:
    print("invalid choice. mean.")
    value = mean

for i in range(len(df)):
    if pd.isnull(df["age"][i]) == True:
        df["age"][i] = value

print(df)
