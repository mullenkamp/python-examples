# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:42:42 2018

@author: MichaelEK
"""
import numpy as np
import pandas as pd
from pdsql import mssql
import os
import geopandas as gpd
from shapely.geometry import Point
from hydrolm.lm import LM
from hydrolm import util
from seaborn import regplot
import matplotlib.pyplot as plt
from gistools.vector import sel_sites_poly

plt.ioff()


############################################
### Parameters to modify

recent_date = '2019-01-01'
min_date = '2004-07-01'
min_count = 10
search_dis = 50000

input_sites = None # None or a list of sites

export_dir = r'E:\ecan\shared\projects\gw_regressions'
fig_sub_dir = 'plots_to_manual'
export_summ1 = 'manual_to_manual_2019-07-10.csv'

############################################
### Other Parameters

server = 'edwprod01'
database = 'hydro'
ts_daily_table = 'TSDataNumericDaily'
ts_hourly_table = 'TSDataNumericHourly'
ts_summ_table = 'TSDataNumericDailySumm'
sites_table = 'ExternalSite'

man_datasets = [13]

qual_codes = [200, 400, 500, 520, 600]

############################################
### Extract summary data and determine the appropriate sites to use

man_summ_data = mssql.rd_sql(server, database, ts_summ_table, where_in={'DatasetTypeID': man_datasets}).drop('ModDate', axis=1)
man_summ_data.FromDate = pd.to_datetime(man_summ_data.FromDate)
man_summ_data.ToDate = pd.to_datetime(man_summ_data.ToDate)

man_sites1 = man_summ_data[(man_summ_data.ToDate >= recent_date) & (man_summ_data.Count >= min_count)].copy()
man_sites1.ExtSiteID = man_sites1.ExtSiteID.str.strip()


###########################################
### Get site data

site_xy = mssql.rd_sql(server, database, sites_table, ['ExtSiteID', 'ExtSiteName', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': man_sites1.ExtSiteID.tolist()})

geometry = [Point(xy) for xy in zip(site_xy['NZTMX'], site_xy['NZTMY'])]
site_xy1 = gpd.GeoDataFrame(site_xy['ExtSiteID'], geometry=geometry, crs=2193).set_index('ExtSiteID')

###########################################
### Iterate through the low flow sites

if isinstance(input_sites, list):
    man_sites2 = man_sites1[man_sites1.ExtSiteID.isin(input_sites)]
else:
    man_sites2 = man_sites1

results_dict = {}
for g in man_sites1.ExtSiteID.tolist():
    print(g)

    man_loc = site_xy1.loc[[g.strip()], :]
    man_loc['geometry'] = man_loc.buffer(search_dis)
    near_rec_sites = sel_sites_poly(site_xy1, man_loc)
    near_rec_sites = near_rec_sites.loc[near_rec_sites.index != g]
    print('There are ' + str(len(near_rec_sites)) + ' sites within range')

    if not near_rec_sites.empty:
        ## Extract all ts data
        g_data = mssql.rd_sql(server, database, ts_daily_table, ['ExtSiteID', 'DateTime', 'Value'], where_in={'DatasetTypeID': man_datasets, 'ExtSiteID': [g], 'QualityCode': qual_codes}, from_date=min_date, date_col='DateTime')
        g_data.DateTime = pd.to_datetime(g_data.DateTime)

        r_data = mssql.rd_sql(server, database, ts_daily_table, ['ExtSiteID', 'DateTime', 'Value'], where_in={'DatasetTypeID': man_datasets, 'ExtSiteID': near_rec_sites.index.tolist(), 'QualityCode': qual_codes}, from_date=min_date, date_col='DateTime')
        r_data.DateTime = pd.to_datetime(r_data.DateTime)

        ## Re-organise the datasets
        g_data1 = g_data.pivot_table('Value', 'DateTime', 'ExtSiteID')
        r_data1 = r_data.pivot_table('Value', 'DateTime', 'ExtSiteID')

        ## Interpolate to fill in missing data
        r_data2 = util.tsreg(pd.concat([g_data1, r_data1], axis=1).drop(g, axis=1), 'D')
        r_data3 = r_data2.interpolate('time', limit=40, limit_area='inside')

        ## Filter the recorder data by guagings
        set1 = pd.concat([g_data1, r_data3], join='inner', axis=1)
        if len(set1) < min_count:
            continue

        ## regressions!
        lm1 = LM(set1.loc[:, (set1.columns != g)], set1[[g]])
        ols1 = lm1.predict('ols', 1, x_transform=None, y_transform=None, min_obs=min_count)

        ## Save
        results_dict.update({g: ols1})


### Produce summary table

cols = ['site', 'nrmse', 'Adj R2', 'nobs', 'y range', 'f value', 'f p value', 'dep sites', 'reg params']

res_list_df = []

res_site = pd.DataFrame()
res_list = []
for i in results_dict:
    model1 = results_dict[i]
    if model1 is not None:
        if model1.sm_xy:
            nrmse1 = model1.nrmse()[i]
            adjr2 = round(model1.sm[i].rsquared_adj, 3)
            nobs = model1.sm[i].nobs
            y_range = model1.sm_xy[i]['y_orig'].max() - model1.sm_xy[i]['y_orig'].min()
            dep_sites = model1.sm_xy[i]['x_orig'].columns.tolist()
            fvalue = round(model1.sm[i].fvalue, 3)
            fpvalue = round(model1.sm[i].f_pvalue, 3)
            params1 = model1.sm[i].params.round(5).tolist()

            site_res = [i, nrmse1, adjr2, nobs, y_range, fvalue, fpvalue, dep_sites, params1]
            res_list.append(site_res)

            ## Plots
            fig = model1.plot_fit(i, dep_sites[0])
            fig.savefig(os.path.join(export_dir, fig_sub_dir, i.replace('/', '-') + '.png'), bbox_inches='tight')
            plt.close(fig)

res_site1 = pd.DataFrame(res_list, columns=cols).set_index('site')


### Save data
file_path = os.path.join(export_dir, export_summ1)
res_site1.to_csv(file_path)




##################################################
### Testing



