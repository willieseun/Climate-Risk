import torch
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn import Softplus
from keras.layers import Dense
import torch.nn.functional as F
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
X_train = np.reshape(X_train, (X_train.shape[0], 6, 6))
X_test = np.reshape(X_test, (X_test.shape[0], 6, 6))

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()
#X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size=0.9, random_state=7)
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)
X.to(device)
y.to(device)
X_train.to(device)
Y_train.to(device)
X_test.to(device)
Y_test.to(device)
print('Done Splitting')
print(X_train.shape)
print(Y_train.shape)
# Define the CNN architecture
#, activity_regularizer=l1(0.001), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_constraint=max_norm(3), bias_constraint=max_norm(3)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def checkpoint(model, filename):
	torch.save(model.state_dict(), filename)
	
def resume(model, filename):
	model.load_state_dict(torch.load(filename))

class LSTMModel(nn.Module):
	def __init__(self):
		super(LSTMModel, self).__init__()
		self.lstm1 = nn.LSTM(6, 160)
		self.lstm2 = nn.LSTM(160, 76)
		self.lstm3 = nn.LSTM(76, 32)
		self.dense1 = nn.Linear(32, 18)
		self.dense2 = nn.Linear(18, 18)

	def forward(self, x):
		x, _ = self.lstm1(x)
		x, _ = self.lstm2(x)
		x, _ = self.lstm3(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x

model = LSTMModel()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
loss_fn = nn.L1Loss()
n_epochs = 6000

early_stop_thresh = 150
best_loss = 1000000
best_epoch = -1

y_pred_list = []
targets_list = []

for epoch in range(n_epochs):
	break
	model.train()
	total_loss = 0.0

	# Iterate over training batches
	for inputs, targets in train_loader:
		optimizer.zero_grad()
		y_pred = model(inputs.to(device))
		y_pred = torch.mean(y_pred, dim=1)
		#print("train_y_pred", y_pred.shape)
		#print("train_targets", targets.shape)		
		loss = loss_fn(y_pred, targets.to(device))
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

	# Calculate average training loss for the epoch
	average_train_loss = total_loss / len(train_loader)
	print(average_train_loss)

	# Validation loop
	model.eval()
	acc = 0
	count = 0

	# Iterate over validation batches
	for inputs, targets in test_loader:
		#targets = targets.unsqueeze(0)
		#inputs = inputs.unsqueeze(0)
		y_pred = model(inputs.to(device))
		#y_pred = y_pred.unsqueeze(1)
		#targets = targets.unsqueeze(1)
		y_pred = y_pred.tolist()
		targets = targets.tolist()
		for y_p in y_pred:
			y_pred_list.append(y_p)
		for tar in targets:
			targets_list.append(tar)
	prd_array = np.array(y_pred_list)
	targets_array = np.array(targets_list)
	prd_array = torch.from_numpy(prd_array)
	prd_array = torch.mean(prd_array, dim=1)

	targets_array = torch.from_numpy(targets_array)
	#print("prd_array", prd_array)
	#print("targets_array", targets_array)
	loss = loss_fn(prd_array, targets_array)
	print("Epoch %d: model loss %.3f" % (epoch, loss))
	if loss < best_loss:
		best_loss = loss
		best_epoch = epoch
		checkpoint(model, "C:\\Users\\willi\\Desktop\\Climate Research\\Data Estimation\\clim_lstm_latest.pth")
	elif epoch - best_epoch > early_stop_thresh:
		print("Early stopped training at epoch %d" % epoch)
		break  # terminate the training loop

resume(model, "C:\\Users\\willi\\Desktop\\Climate Research\\Data Estimation\\clim_lstm_latest.pth")
print(model)

y_pred = model(X_test.to(device))
y_pred = y_pred.cpu().detach().numpy()
y_pred = np.mean(y_pred, axis=1)
Y_test = Y_test.cpu().detach().numpy()
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
#df.to_csv('shifted.csv', index=False)
#X_rs = datadf.drop(columns=["state", "state_clim", "state_soil"]).values
#X_rs = datadf[["cfips", "cfips_norm", "year", "year_norm"]].values
X_rs = np.concatenate((years, cfips, year_norm, cfips_norm, sta_scal_years, sta_scal_cfips, minmax_scal_years, minmax_scal_cfips, rob_scal_years, rob_scal_cfips, max_scal_years, max_scal_cfips, pow_scal_years, pow_scal_cfips, qua_scal_years, qua_scal_cfips, norm_scal_years, norm_scal_cfips), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler.fit_transform(X_rs)
X_rs = np.hstack((X_rs, scaler1))
#X_rs = scaler1
X_rs = np.reshape(X_rs, (X_rs.shape[0], 6, 6))
X_rs = torch.from_numpy(X_rs).float()
Y_rs = model(X_rs.to(device))
lst = ["PS", "TS", "T2M", "QV2M", "RH2M", "WS2M", "WS10M", "GWETTOP", "T2M_MAX", "T2M_MIN", "GWETPROF", "GWETROOT", "PRECTOTCORR", "ALLSKY_SRF_ALB", "PRECTOTCORR_SUM", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "CLRSKY_SFC_PAR_TOT"]
Y_rs = Y_rs.cpu().detach().numpy()
Y_rs = np.mean(Y_rs, axis=1)
prd_df[lst] = Y_rs
#prd_df["PRECTOTCORR_SUM"] = prd_df["PRECTOTCORR_SUM"]*1000
prd_df.to_csv('Future Climate LSTM.csv', index=False)