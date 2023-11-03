import torch
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from keras.layers import Dense
import torch.nn.functional as F
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, Normalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# fix random seed for reproducibility
np.random.seed(42)
batch_size = 16
device = torch.device("cuda:0")

standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
robust_scaler = RobustScaler()
max_abs_scaler = MaxAbsScaler()
power_transformer = PowerTransformer(method='box-cox')
quantile_transformer = QuantileTransformer(output_distribution='uniform')
normalizer = Normalizer()

# Read csv file
datadf = pd.read_csv('Final Data Clean.csv')
datadf = datadf.dropna()
soildf = pd.read_csv('Soil_data.csv')
#soildf = soildf.drop(columns=["state"])
df = pd.merge(datadf, soildf, on=['cfips'])
datadf = pd.merge(datadf, soildf, on=['cfips'])
for col in df.columns:
	shf = "_shifted"
	df[col+shf] = df[col].shift(1)

df.replace('', np.nan, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
df = df.interpolate(method='linear')
# Iterate over the columns in the DataFrame and interpolate missing values
for col in df.columns:
    df[col].fillna(method='ffill', inplace=True)
    df[col].fillna(method='bfill', inplace=True)
print(df)
#df.to_csv('shifted.csv', index=False)
X = datadf.drop(columns=["state", "state_soil", "MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]).values
y = datadf[["MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit_transform(X)
X = np.hstack((X, scaler))
print(scaler.shape)
print(y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X.astype(np.float32), y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
# Calculate the MSE and RMSE
mae = mean_absolute_error(Y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(Y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
r2s = r2_score(Y_test, y_pred, multioutput='raw_values')
# Print the MAE and MSE
print('Mean absolute error:', mae)
print('Mean squared error:', mse)
print('Root Mean squared error:', rmse)
print('R squared error:', r2s)

# Read csv file
prd_df = pd.DataFrame()
datadf = pd.read_csv('Future Climate CNN_N.csv')
soildf = pd.read_csv('Soil_data.csv')
datadf = pd.merge(datadf, soildf, on=['cfips'])
datadf = datadf.dropna()
prd_df['cfips'] = datadf['cfips']
prd_df['state'] = datadf['state']
prd_df['year'] = datadf['year']
print(df)
X_rs = datadf.drop(columns=["state", "state_soil"]).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler.fit_transform(X_rs)
X_rs = np.hstack((X_rs, scaler1))
lst = ["MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]
Y_rs = model.predict(X_rs)
prd_df[lst] = Y_rs
prd_df.to_csv('Future Maize Production RF.csv', index=False)
