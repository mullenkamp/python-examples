# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:28:28 2018

@author: MichaelEK
"""
import pandas as pd
import geopandas as gpd
from gistools import vector, util
from gistools.datasets import get_path

pd.options.display.max_columns = 10

####################################
### Parameters

sites_shp = 'flow_recorders_pareora'
rec_streams_shp = 'rec_streams_pareora'
rec_catch_shp = 'rec_catch_pareora'
catch_shp = 'catchment_pareora'

sites_shp_path = get_path(sites_shp)
rec_streams_shp_path = get_path(rec_streams_shp)
rec_catch_shp_path = get_path(rec_catch_shp)
catch_shp_path = get_path(catch_shp)

sites_col_name = 'SITENUMBER'
poly_col_name = 'Catchmen_1'
line_site_col = 'NZREACH'

#######################################
### Examples

pts = util.load_geo_data(sites_shp_path)
pts['geometry'] = pts.geometry.simplify(1)

## Selecting points from within a polygon
pts1 = vector.sel_sites_poly(sites_shp_path, rec_catch_shp_path, buffer_dis=10)

## Spatial join attributes of polygons to points
pts2, poly2 = vector.pts_poly_join(sites_shp_path, catch_shp_path, poly_col_name)

## Create a GeoDataFrame from x and y data
pts_df = pts[[sites_col_name, 'geometry']].copy()
pts_df['x'] = pts_df.geometry.x
pts_df['y'] = pts_df.geometry.y
pts_df.drop('geometry', axis=1, inplace=True)

pts3 = vector.xy_to_gpd(sites_col_name, 'x', 'y', pts_df)

## Find the closest line to points
line1 = vector.closest_line_to_pts(sites_shp_path, rec_streams_shp_path, line_site_col, buffer_dis=100)


