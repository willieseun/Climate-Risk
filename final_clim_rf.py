import torch
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn import Softplus
from keras.layers import Dense
import torch.nn.functional as F
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
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
climdf = pd.read_csv('stat_clim.csv')
soildf = pd.read_csv('Soil_data.csv')
df = pd.merge(datadf, soildf, on=['cfips'])
df = pd.merge(df, climdf, on=['cfips'])
datadf = pd.merge(datadf, soildf, on=['cfips'])
datadf = pd.merge(datadf, climdf, on=['cfips'])
#datadf["PRECTOTCORR_SUM"] = datadf["PRECTOTCORR_SUM"]/1000
cfips = datadf[["cfips"]].values
years = datadf[["year"]].values
sta_scal_years = standard_scaler.fit_transform(years)
sta_scal_cfips = standard_scaler.fit_transform(cfips)
minmax_scal_years = min_max_scaler.fit_transform(years)
minmax_scal_cfips = min_max_scaler.fit_transform(cfips)
rob_scal_years = robust_scaler.fit_transform(years)
rob_scal_cfips = robust_scaler.fit_transform(cfips)
max_scal_years = max_abs_scaler.fit_transform(years)
max_scal_cfips = max_abs_scaler.fit_transform(cfips)
pow_scal_years = power_transformer.fit_transform(years)
pow_scal_cfips = power_transformer.fit_transform(cfips)
qua_scal_years = quantile_transformer.fit_transform(years)
qua_scal_cfips = quantile_transformer.fit_transform(cfips)
norm_scal_years = normalizer.fit_transform(years)
norm_scal_cfips = normalizer.fit_transform(cfips)
cfips_norm = (cfips - 1000) / 1036
year_norm = (years - 1980) / 2520
datadf['cfips_norm'] = cfips_norm
datadf['year_norm'] = year_norm
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
#X = datadf.drop(columns=["state", "state_soil", "state_clim", "MAIZE_PRODUCTION(MT)", "LAND_HECTARES", "PS", "TS", "T2M", "QV2M", "RH2M", "WS2M", "WS10M", "GWETTOP", "T2M_MAX", "T2M_MIN", "GWETPROF", "GWETROOT", "PRECTOTCORR", "ALLSKY_SRF_ALB", "PRECTOTCORR_SUM", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "CLRSKY_SFC_PAR_TOT"]).values
#X = datadf[["cfips", "cfips_norm", "year", "year_norm"]].values
X = np.concatenate((years, cfips, year_norm, cfips_norm, sta_scal_years, sta_scal_cfips, minmax_scal_years, minmax_scal_cfips, rob_scal_years, rob_scal_cfips, max_scal_years, max_scal_cfips, pow_scal_years, pow_scal_cfips, qua_scal_years, qua_scal_cfips, norm_scal_years, norm_scal_cfips), axis=1)
y = df[["PS", "TS", "T2M", "QV2M", "RH2M", "WS2M", "WS10M", "GWETTOP", "T2M_MAX", "T2M_MIN", "GWETPROF", "GWETROOT", "PRECTOTCORR", "ALLSKY_SRF_ALB", "PRECTOTCORR_SUM", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "CLRSKY_SFC_PAR_TOT"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit_transform(X)
X = np.hstack((X, scaler))
#X = scaler
print(scaler.shape)
print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X.astype(np.float32), y, test_size=0.2, random_state=42)


model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# Make predictions on the test set
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
datadf = pd.read_csv('Future_climate.csv')
climdf = pd.read_csv('stat_clim.csv')
soildf = pd.read_csv('Soil_data.csv')
prd_df['cfips'] = datadf['cfips']
prd_df['state'] = datadf['state']
prd_df['year'] = datadf['year']
df = pd.merge(datadf, soildf, on=['cfips'])
df = pd.merge(df, climdf, on=['cfips'])
datadf = pd.merge(datadf, soildf, on=['cfips'])
datadf = pd.merge(datadf, climdf, on=['cfips'])
cfips = datadf[["cfips"]].values
years = datadf[["year"]].values
sta_scal_years = standard_scaler.fit_transform(years)
sta_scal_cfips = standard_scaler.fit_transform(cfips)
minmax_scal_years = min_max_scaler.fit_transform(years)
minmax_scal_cfips = min_max_scaler.fit_transform(cfips)
rob_scal_years = robust_scaler.fit_transform(years)
rob_scal_cfips = robust_scaler.fit_transform(cfips)
max_scal_years = max_abs_scaler.fit_transform(years)
max_scal_cfips = max_abs_scaler.fit_transform(cfips)
pow_scal_years = power_transformer.fit_transform(years)
pow_scal_cfips = power_transformer.fit_transform(cfips)
qua_scal_years = quantile_transformer.fit_transform(years)
qua_scal_cfips = quantile_transformer.fit_transform(cfips)
norm_scal_years = normalizer.fit_transform(years)
norm_scal_cfips = normalizer.fit_transform(cfips)
cfips_norm = (cfips - 1000) / 1036
year_norm = (years - 1980) / 2520
datadf['cfips_norm'] = cfips_norm
datadf['year_norm'] = year_norm
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
print(datadf.shape)
X_rs = np.concatenate((years, cfips, year_norm, cfips_norm, sta_scal_years, sta_scal_cfips, minmax_scal_years, minmax_scal_cfips, rob_scal_years, rob_scal_cfips, max_scal_years, max_scal_cfips, pow_scal_years, pow_scal_cfips, qua_scal_years, qua_scal_cfips, norm_scal_years, norm_scal_cfips), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler.fit_transform(X_rs)
X_rs = np.hstack((X_rs, scaler1))
lst = ["PS", "TS", "T2M", "QV2M", "RH2M", "WS2M", "WS10M", "GWETTOP", "T2M_MAX", "T2M_MIN", "GWETPROF", "GWETROOT", "PRECTOTCORR", "ALLSKY_SRF_ALB", "PRECTOTCORR_SUM", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "CLRSKY_SFC_PAR_TOT"]
Y_rs = model.predict(X_rs)
prd_df[lst] = Y_rs
prd_df.to_csv('Future Climate RF.csv', index=False)
