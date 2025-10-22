import pandas as pd


data = {'colour': ['red', 'blue', 'green', 'red', 'blue']}
df = pd.DataFrame(data)
print("Original Data:")
print(df)

unique=df["colour"].unique()
for val in unique:
    temp_column=[]
    for i in df['colour']:
        if i==val:
            temp_column.append(1)
        else:
            temp_column.append(0)
    df[f"colour_{val}"]=temp_column
print(df)
