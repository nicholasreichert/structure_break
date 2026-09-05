import pandas as pd

X_train = pd.read_parquet("data\X_train.parquet")
X_test = pd.read_parquet("data\X_test.reduced.parquet")

y_train = pd.read_parquet("data\y_train.parquet")
y_test = pd.read_parquet("data\y_test")