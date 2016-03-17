import numpy as np
import scipy.io as sio
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
import scipy.io as sio
#from Scientific.IO.NetCDF import NetCDFFile
#from glob import glob
#from scipy.interpolate import griddata
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

year=2009

base_dir='../../../../Data/YOTC/'

mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}

#temperatura
#******************************************************************************
file = Dataset(base_dir+'yotc_temp.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

lats  = file.variables['latitude'][:] #(240,0)
lons  = file.variables['longitude'][:] #(121,0)
# Find the nearest latitude and longitude for Darwin
lat_idx = np.abs(lats - mac['lat']).argmin()
lon_idx = np.abs(lons - mac['lon']).argmin()
level  = file.variables['level'][:] #(20,0)
time  = file.variables['time'][:] #(1460,)
temp = file.variables['t'][:,:,lat_idx ,lon_idx] #(1460, 20, 121, 240)
t_units = file.variables['t'].units
print t_units
file.close

#wind u
#******************************************************************************
file = Dataset(base_dir+str(year)+'_u.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

u = file.variables['u'][:,:,lat_idx ,lon_idx] #(1460, 20, 121, 240)
u_units = file.variables['u'].units
print u_units
file.close

#*****************************************************************************\******************************************************************************
#Write mat-file MAC

sio.savemat('yotc_mac2008-2010.mat', {'temp':temp, 'time':time})
