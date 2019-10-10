# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:15:54 2019

@author: michaelek
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp import Interp
from pyproj import Proj, transform
from datetime import datetime
from pdsql import mssql

pd.options.display.max_columns = 10


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

########################################
### Parameters

dataset_type = [15]

server = 'edwprod01'
db = 'hydro'


py_dir = os.path.realpath(os.path.dirname(__file__))

point_time_name = 'DateTime'
point_x_name = 'NZTMX'
point_y_name = 'NZTMY'
point_data_name = 'Value'
point_crs = 2193

from_date = '2019-01-01'
to_date = '2019-01-31'
grid_res = 5000
to_crs = 2193
method = 'linear' # or 'cubic'
min_val = 0

########################################
### Get data

## Met station data
summ1 = mssql.rd_sql(server, db, 'TSDataNumericDailySumm', where_in={'DatasetTypeID': dataset_type}).drop('ModDate', axis=1)
summ1.ToDate = pd.to_datetime(summ1.ToDate)
summ1.FromDate = pd.to_datetime(summ1.FromDate)

summ2 = summ1[(summ1.FromDate <= from_date) & (summ1.ToDate >= from_date)].copy()

ts_data = mssql.rd_sql(server, db, 'TSDataNumericDaily', ['ExtSiteID', 'DateTime', 'Value'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist(), 'DatasetTypeID': dataset_type}, from_date=from_date, to_date=to_date, date_col='DateTime')
ts_data.DateTime = pd.to_datetime(ts_data.DateTime)
ts_data = ts_data[ts_data.ExtSiteID.astype(int) < 900000].copy()

## Site data
site_data = mssql.rd_sql(server, db, 'ExternalSite', ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist()}).round()

## Combine ts data and site locations
point_data = pd.merge(ts_data, site_data, on='ExtSiteID').drop('ExtSiteID', axis=1)

######################################
### Interpolations

i1 = Interp(point_data=point_data, point_time_name=point_time_name, point_x_name=point_x_name, point_y_name=point_y_name, point_data_name=point_data_name, point_crs=point_crs)

new_grid = i1.points_to_grid(grid_res, to_crs, method=method, min_val=min_val)

## Convert to DataFrame
new_df = new_grid.precip.to_dataframe().dropna()


## Plots
new_grid.precip.isel(time=0).plot()

