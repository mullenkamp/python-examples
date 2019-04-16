# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:05:46 2018

@author: MichaelEK
"""
from hilltoppy.web_service import measurement_list, site_list, get_data, wq_sample_parameter_list

###########################################
### Parameters
base_url = 'http://wateruse.ecan.govt.nz'
hts = 'WQAll.hts'
site = 'SQ31045'
measurement = 'Total Phosphorus'
from_date = '1983-11-22 10:50'
to_date = '2018-04-13 14:05'
dtl_method = 'trend'

##########################################
### Examples

## Get site list
sites = site_list(base_url, hts)

## Get the measurement types for a specific site
mtype_df1 = measurement_list(base_url, hts, site)

## Get the water quality parameter data (only applies to WQ data)
mtype_df2 = wq_sample_parameter_list(base_url, hts, site)

## Get the time series data for a specific site and measurement type
tsdata1 = get_data(base_url, hts, site, measurement, from_date=from_date, to_date=to_date)

## Get extra WQ time series data (only applies to WQ data)
tsdata2, extra2 = get_data(base_url, hts, site, measurement, from_date=from_date, to_date=to_date, parameters=True)

## Get WQ sample data (only applies to WQ data)
tsdata3 = get_data(base_url, hts, site, 'WQ Sample', from_date=from_date, to_date=to_date)

## Convert values under the detection limit to numeric values (only applies to WQ data)
tsdata4, extra4 = get_data(base_url, hts, site, measurement, from_date=from_date, to_date=to_date, parameters=True, dtl_method=dtl_method)

