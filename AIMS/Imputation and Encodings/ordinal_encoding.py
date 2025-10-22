import pandas as pd


dataset={"size":["small","large","medium","medium","small","large"]}
data=pd.DataFrame(dataset)
order=["small","medium","large"]
print(data)
ordinal_data=[]
for x in data["size"]:
    ordinal_data.append(order.index(x))
data["ordinal"]=ordinal_data
print(data)