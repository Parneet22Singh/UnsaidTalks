#Numpy Array Exercise
import numpy as np
arr=np.random.rand(4,4)
print(arr)
print(f"Second row is {arr[1]}")

#CSV Data Analysis
import kagglehub
import pandas as pd
path = kagglehub.dataset_download("anthonypino/melbourne-housing-market")
print("Path to dataset files:", path)
df = pd.read_csv("/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv")
df.head(5)

#DataFrame Calculation
#Since Revenue doesn't exist we will use Price column instead
df['Sales_Tax'] = df['Price'] * 0.05
print(df[['Price', 'Sales_Tax']].head())

#Conditional Data FIltering
x=df[df["Landsize"]<190]
#x=df[df["Landsize"]<190].head(10)
print(x)
print(x.shape)
