import numpy as np
import scipy.io as sio
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
import scipy.io as sio
import os
from Scientific.IO.NetCDF import NetCDFFile
#from glob import glob
#from scipy.interpolate import griddata
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/BLH/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Scripts/Python/ERA-I/BLH/'
mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}
#******************************************************************************

file = Dataset(path_data+ '2010_blh.nc', 'r')

lats  = file.variables['latitude'][:] #(240,)
lons  = file.variables['longitude'][:] #(121,)
lat_idx = np.abs(lats - mac['lat']).argmin()
lon_idx = np.abs(lons - mac['lon']).argmin()
time  = file.variables['time'][:] #every 12 h
blh = file.variables['blh'][:,lat_idx ,lon_idx]
blh_units = file.variables['blh'].units
file.close


print '----------------------'




#*****************************************************************************\******************************************************************************
#Write mat-file MAC

sio.savemat(path_data_save+ 'blh_erai_mac_2010.mat', {'time':time,'blh':blh, 'blh_units':blh_units})


print '*************************************************'
