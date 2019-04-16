# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:47:54 2018

@author: MichaelEK
"""
import os
import pandas as pd
from pdsql import mssql
from allotools import AlloUsage
from datetime import datetime

pd.options.display.max_columns = 10


############################################
### Parameters

server = 'edwprod01'
database = 'hydro'
sites_table = 'ExternalSite'

catch_group = ['Ashburton River']
summ_col = 'SwazName'

crc_filter = {'use_type': ['stockwater', 'irrigation']}

datasets = ['allo', 'metered_allo', 'restr_allo', 'metered_restr_allo', 'usage']

freq = 'A-JUN'

from_date = '2012-07-01'
to_date = '2018-06-30'

py_path = os.path.realpath(os.path.dirname(__file__))

plot_dir = 'plots'
export2 = 'swaz_allo_usage_2019-03-25.csv'
export3 = 'swaz_allo_usage_pivot_2019-03-25.csv'

now1 = str(datetime.now().date())

plot_path = os.path.join(py_path, plot_dir)

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

############################################
### Extract data

sites1 = mssql.rd_sql(server, database, sites_table, ['ExtSiteID', 'SwazGroupName', summ_col], where_in={'SwazGroupName': catch_group})

site_filter = {'SwazName': sites1.SwazName.unique().tolist()}

a1 = AlloUsage(from_date, to_date, site_filter=site_filter, crc_filter=crc_filter)

combo_ts = a1.get_ts(datasets, freq, ['SwazName', 'use_type', 'date'], irr_season=True)

combo_ts.to_csv(os.path.join(py_path, export2))


#########################################
### Plotting

### Grouped
## Lumped
a1.plot_group('A-JUN', group='SwazGroupName', export_path=plot_path, irr_season=True)

## broken up
a1.plot_group('A-JUN', group='SwazName', export_path=plot_path, irr_season=True)

### Stacked
## lumped
a1.plot_stacked('A-JUN', group='SwazGroupName', export_path=plot_path, irr_season=True)

## broken up
a1.plot_stacked('A-JUN', group='SwazName', export_path=plot_path, irr_season=True)


