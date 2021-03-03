"""
Created by Mike Kittridge on 2020-09-01.
Contains the code to train and test the flood forecast model.

"""
import os
import numpy as np
import pandas as pd
import requests
import orjson
import zstandard as zstd
import xarray as xr
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
# from sklearn.inspection import permutation_importance
from scipy.signal import argrelextrema
from pprint import pprint
import pickle
# %matplotlib inline

pd.options.display.max_columns = 10

#####################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

base_url = 'http://tethys-ts.xyz/tethys/data/'

# precip_sites = ['217810', '218810', '219510', '219910', '228213', '310510', '311810', '311910', '320010', '321710']
flow_sites = ['66401']
wl_sites = ['66402']

long_precip = ['219510', '219910', '320010']

from_date = '1980-07-01T00:00'
to_date = '2019-07-1T00:00'

train_date_cut_off = '2010-07-01'

n_hours_shift_start = 14
n_hours_shift_end = 48

model_file1 = 'waimak_flood_model_66401_streamflow_v03.skl.pkl'

dc = zstd.ZstdDecompressor()

####################################
### Get data

## Datasets
datasets = requests.get(base_url + 'get_datasets').json()

p_dataset = [d for d in datasets if (d['feature'] == 'atmosphere') and
                                    (d['parameter'] == 'precipitation') and
                                    (d['product_code'] == 'quality_controlled_data') and
                                    (d['owner'] == 'Environment Canterbury')][0]
f_dataset = [d for d in datasets if (d['feature'] == 'waterway') and
                                    (d['parameter'] == 'streamflow') and
                                    (d['product_code'] == 'quality_controlled_data') and
                                    (d['owner'] == 'Environment Canterbury')][0]

pprint(p_dataset, sort_dicts=False)
pprint(f_dataset, sort_dicts=False)

## Sites
p_sites1 = requests.post(base_url + 'get_stations', params={'dataset_id': p_dataset['dataset_id'], 'compression': 'zstd'})
p_sites2 = orjson.loads(dc.decompress(p_sites1.content))
p_sites = [p for p in p_sites2 if p['ref'] in long_precip]

f_sites1 = requests.post(base_url + 'get_stations', params={'dataset_id': f_dataset['dataset_id'], 'compression': 'zstd'})
f_sites2 = orjson.loads(dc.decompress(f_sites1.content))
f_sites = [f for f in f_sites2 if f['ref'] in flow_sites][0]

pprint(p_sites, sort_dicts=False)
pprint(f_sites, sort_dicts=False)

###########################################33
### TS Data processing

## Precip Data

precip_r_dict = {}
for p in p_sites:
    print(p['ref'])
    r = requests.get(base_url + 'get_results', params=
                     {'dataset_id': p_dataset['dataset_id'],
                      'station_id': p['station_id'],
                      'compression': 'zstd', 'from_date': from_date,
                      'to_date': to_date,
                      'remove_height': True})
    xr1 = xr.DataArray.from_dict(orjson.loads(dc.decompress(r.content)))
    xr1.name = 'precip'
    df1 = xr1.to_dataframe()
    df1.index = pd.to_datetime(df1.index) + pd.DateOffset(hours=12)
    precip_r_dict.update({p['ref']: df1.copy()})

p_list = []
for s, df1 in precip_r_dict.items():
    df2 = df1['precip'].resample('H').sum().iloc[1:-1].fillna(0)
    site_name = s
    df_list = []
    for d in range(n_hours_shift_start, n_hours_shift_end+1):
        n1 = df2.shift(d, 'H')
        n1.name = site_name + '_' + str(d)
        df_list.append(n1)
    df4 = pd.concat(df_list, axis=1).dropna()

    p_list.append(df4)

p_data = pd.concat(p_list, axis=1).dropna()
print(p_data)

p_list = []
for s, df1 in precip_r_dict.items():
    df2 = df1['precip'].resample('D').sum().iloc[1:-1].fillna(0)
    site_name = s
    df_list = []
    for d in range(0, 2+1):
        n1 = df2.shift(d, 'D')
        n1.name = site_name + '_' + str(d)
        df_list.append(n1)
    df4 = pd.concat(df_list, axis=1).dropna()

    p_list.append(df4)

p_data_daily = pd.concat(p_list, axis=1).dropna()
print(p_data_daily)

## Flow data
r = requests.get(base_url + 'get_results', params=
                 {'dataset_id': f_dataset['dataset_id'],
                  'station_id': f_sites['station_id'],
                  'compression': 'zstd',
                  'from_date': from_date,
                  'to_date': to_date,
                  'remove_height': True})
xr1 = xr.DataArray.from_dict(orjson.loads(dc.decompress(r.content)))
xr1.name = '66401_0'
df1 = xr1.to_dataframe()
df1.index = pd.to_datetime(df1.index) + pd.DateOffset(hours=12)

f_data = df1.copy()
print(f_data)

f_data_daily = f_data.resample('D').mean().iloc[1:-1]

###############################################
### Daily model

## Prepare data
label_name = '66401_0'
actual = f_data_daily[label_name].loc[train_date_cut_off:]
actual.name = 'Actual Flow'

data1 = pd.concat([f_data_daily, p_data_daily], axis=1).dropna()

train_features_df = data1.loc[:train_date_cut_off].drop(label_name, axis = 1)
train_labels = np.array(data1.loc[:train_date_cut_off, label_name])
train_features = np.array(train_features_df)

test_features_df = data1.loc[train_date_cut_off:].drop(label_name, axis = 1)

test_features = np.array(test_features_df)
test_labels = np.array(actual)

print(train_labels)
print(train_labels.shape)
print("")
print(train_features)
print(train_features.shape)

## Train model
rf = GradientBoostingRegressor(n_estimators = 100)
rf.fit(train_features, train_labels)

# rf = RandomForestRegressor(n_estimators = 200)
# rf.fit(train_features, train_labels)

## Make the predictions and combine with the actuals
predictions1 = rf.predict(test_features)
predict1 = pd.Series(predictions1, index=test_features_df.index, name='GB Predicted Flow (m^3/s)')

combo1 = pd.merge(actual.reset_index(), predict1.reset_index(), how='left').set_index('time')

print(combo1)

## Check importances

# Get numerical feature importances -- Must be run without the HistGB
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(train_features_df.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))


###############################################
### Houly model

### Prepare data
label_name = '66401_0'
actual = f_data[label_name].loc[train_date_cut_off:]
actual.name = 'Actual Flow'

data1 = pd.concat([f_data, p_data], axis=1).dropna()

train_features_df = data1.loc[:train_date_cut_off].drop(label_name, axis = 1)
train_labels = np.array(data1.loc[:train_date_cut_off, label_name])
train_features = np.array(train_features_df)

test_features_df = data1.loc[train_date_cut_off:].drop(label_name, axis = 1)

test_features = np.array(test_features_df)
test_labels = np.array(actual)

print(train_labels)
print(train_labels.shape)
print("")
print(train_features)
print(train_features.shape)

## Train model
rf = GradientBoostingRegressor(n_estimators = 100)
rf.fit(train_features, train_labels)

# rf = RandomForestRegressor(n_estimators = 200)
# rf.fit(train_features, train_labels)

## Make the predictions and combine with the actuals
predictions1 = rf.predict(test_features)
predict1 = pd.Series(predictions1, index=test_features_df.index, name='GB Predicted Flow (m^3/s)')

combo1 = pd.merge(actual.reset_index(), predict1.reset_index(), how='left').set_index('time')

print(combo1)

### Process results
max_index = argrelextrema(test_labels, np.greater, order=12)[0]

upper_index = np.where(test_labels > np.percentile(test_labels, 80))[0]

test_labels_index = max_index[np.in1d(max_index, upper_index)]

max_data = combo1.iloc[test_labels_index]

print(max_data)

## Estimate accuracy/errors
p1 = max_data.iloc[:, 1]
a1 = max_data.iloc[:, 0]

errors = abs(p1 - a1)
bias_errors = (p1 - a1)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'm3/s.')
print('Mean Error (Bias):', round(np.mean(bias_errors), 2), 'm3/s.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / a1)
#
# Calculate and display accuracy
accuracy = np.mean(mape)
print('MANE:', round(accuracy, 2), '%.')

bias1 = np.mean(100 * (bias_errors / a1))
print('MNE:', round(bias1, 2), '%.')

bias2 = 100 * np.mean(bias_errors)/np.mean(a1)
print('NME:', round(bias2, 2), '%.')

# Get numerical feature importances -- Must be run without the Hist
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(train_features_df.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))


## Plotting
ax = combo1.plot(lw=2)
max_data1 = max_data.reset_index().rename(columns={'time': 'Date', 'Actual Flow': 'Flow (m^3/s)'})
max_data1.plot.scatter('Date', 'Flow (m^3/s)', ax=ax, fontsize=15, lw=3)
# plt.show()

max_data2 = max_data1.sort_values('Flow (m^3/s)')
ax = max_data2.set_index('Flow (m^3/s)', drop=False)['Flow (m^3/s)'].plot.line(color='red', lw=2)
max_data2.plot.scatter('Flow (m^3/s)', 'GB Predicted Flow (m^3/s)', ax=ax, fontsize=15, lw=2)
# plt.show()

# print(max_data2)
max_data2 = max_data1.sort_values('Flow (m^3/s)').drop('Date', axis=1)
max_data2 = np.log(max_data2)
ax = max_data2.set_index('Flow (m^3/s)', drop=False)['Flow (m^3/s)'].plot.line(color='red', lw=2)
max_data2.plot.scatter('Flow (m^3/s)', 'GB Predicted Flow (m^3/s)', ax=ax, fontsize=15, lw=2)
# plt.show()

##################################
### Save the model

with open(os.path.join(base_dir, model_file1), 'wb') as f:
    pickle.dump(rf, f)


# with open(os.path.join(base_dir, model_file), 'rb') as f:
#     rff = pickle.load(f)






