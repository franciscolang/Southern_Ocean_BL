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
file = Dataset(base_dir+'2008_temp.nc', 'r')

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
file = Dataset(base_dir+'yotc_u.nc', 'r')

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

#wind v
#******************************************************************************
file = Dataset(base_dir+'yotc_v.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

v = file.variables['v'][:,:,lat_idx ,lon_idx] #(1460, 20, 121, 240)
v_units = file.variables['v'].units
print v_units
file.close

#*****************************************************************************\
#w
#******************************************************************************
file = Dataset(base_dir+'yotc_w.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

w = file.variables['w'][:,:,lat_idx ,lon_idx] #(1460, 20, 121, 240)
w_units = file.variables['w'].units
print w_units
file.close

#*****************************************************************************
#q
#******************************************************************************
file = Dataset(base_dir+'yotc_q.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

q = file.variables['q'][:,:,lat_idx ,lon_idx] #(1460, 20, 121, 240)
q_units = file.variables['q'].units
print q_units
file.close

#*****************************************************************************\
#level generation



#*****************************************************************************\******************************************************************************
#Write mat-file MAC

sio.savemat(base_dir+'mat/yotc_mac2008-2010.mat', {'time':time,'temp':temp, 't_units':t_units,'u':u, 'u_units':u_units,'v':v, 'v_units':v_units,'w':w, 'w_units':w_units,'q':q, 'q_units':q_units })
