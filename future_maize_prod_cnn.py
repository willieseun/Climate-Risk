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
soildf = pd.read_csv('Soil_data.csv')
#soildf = soildf.drop(columns=["state"])
datadf = datadf.dropna()
df = pd.merge(datadf, soildf, on=['cfips'])
datadf = pd.merge(datadf, soildf, on=['cfips'])
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
params_lst = [year_norm, cfips_norm, sta_scal_years, sta_scal_cfips, minmax_scal_years, minmax_scal_cfips, rob_scal_years, rob_scal_cfips, max_scal_years, max_scal_cfips, pow_scal_years, pow_scal_cfips, qua_scal_years, qua_scal_cfips, norm_scal_years, norm_scal_cfips]
count = 0
for params in params_lst:
	prm = str('fts') + str(count)
	datadf[prm] = params
	count += 1
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
viw = datadf.drop(columns=["state", "state_soil", "MAIZE_PRODUCTION(MT)", "LAND_HECTARES"])
print(viw.columns)
X = datadf.drop(columns=["state", "state_soil", "MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]).values
y = df[["MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit_transform(X)
X = np.hstack((X, scaler))
print(X.shape)
print(y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X.astype(np.float32), y, test_size=0.2, random_state=42)
X = np.reshape(X, (X.shape[0], 4, 47))
X_train = np.reshape(X_train, (X_train.shape[0], 4, 47))
X_test = np.reshape(X_test, (X_test.shape[0], 4, 47))

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

class CNN(torch.nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		# Convolutional layers
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		self.flat = nn.Flatten()
		# Batch normalization layer
		self.bn = nn.BatchNorm2d(32)

		# Pooling layer
		self.pool = nn.MaxPool2d(2, 2)

		# Fully connected layers
		self.fc1 = nn.Linear(288, 148)
		self.fc2 = nn.Linear(148, 2)
		self.softplus = Softplus()


	def forward(self, x):
		x = x.unsqueeze(1)
		x = x.float()
		x = x.to(device)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn(x)
		x = self.pool(x)

		# Flatten the tensor before passing it to the fully connected layers
		x = self.flat(x)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.softplus(x)

		return x

model = CNN()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00022)
loss_fn = nn.SmoothL1Loss()
n_epochs = 6000

early_stop_thresh = 1500
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

	targets_array = torch.from_numpy(targets_array)
	#print("prd_array", prd_array)
	#print("targets_array", targets_array)
	loss = loss_fn(prd_array, targets_array)
	print("Epoch %d: model loss %.3f" % (epoch, loss))
	if loss < best_loss:
		best_loss = loss
		best_epoch = epoch
		checkpoint(model, "clim_cnn_best_model_best_path.pth")
	elif epoch - best_epoch > early_stop_thresh:
		print("Early stopped training at epoch %d" % epoch)
		break  # terminate the training loop

resume(model, "clim_cnn_best_model_best_path.pth")
print(model)

y_pred = model(X_test.to(device))
y_pred = y_pred.cpu().detach().numpy()
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
datadf = pd.read_csv('Future Climate CNN_N.csv')
datadf = datadf.dropna()
soildf = pd.read_csv('Soil_data.csv')
datadf = pd.merge(datadf, soildf, on=['cfips'])
prd_df['cfips'] = datadf['cfips']
prd_df['state'] = datadf['state']
prd_df['year'] = datadf['year']
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
params_lst = [year_norm, cfips_norm, sta_scal_years, sta_scal_cfips, minmax_scal_years, minmax_scal_cfips, rob_scal_years, rob_scal_cfips, max_scal_years, max_scal_cfips, pow_scal_years, pow_scal_cfips, qua_scal_years, qua_scal_cfips, norm_scal_years, norm_scal_cfips]
count = 0
for params in params_lst:
	prm = str('fts') + str(count)
	datadf[prm] = params
	count += 1
datadf['cfips_norm'] = cfips_norm
datadf['year_norm'] = year_norm
print(df)
viw = datadf.drop(columns=["state", "state_soil"])
print(viw.columns)
X_rs = datadf.drop(columns=["state", "state_soil"]).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler.fit_transform(X_rs)
X_rs = np.hstack((X_rs, scaler1))
print(X_rs.shape)
X_rs = np.reshape(X_rs, (X_rs.shape[0], 4, 47))
X_rs = torch.from_numpy(X_rs).float()
Y_rs = model(X_rs.to(device))
lst = ["MAIZE_PRODUCTION(MT)", "LAND_HECTARES"]
Y_rs = Y_rs.cpu().detach().numpy()
#print(Y_rs)
prd_df[lst] = Y_rs
prd_df.to_csv('Future Maize Production CNN.csv', index=False)
