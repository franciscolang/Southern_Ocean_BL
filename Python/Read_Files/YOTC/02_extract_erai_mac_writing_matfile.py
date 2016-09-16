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
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/'
path_data='/media/flang/Fco Lang T1/ERA-I/'
mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}

for Y in range(1996,1997):

    file = Dataset(path_data+ str(Y) +'/T_' +str(Y)+ '_'+str(1).zfill(2)+'.nc', 'r')

    lats  = file.variables['g0_lat_2'][:] #(240,)
    lons  = file.variables['g0_lon_3'][:] #(121,)
    lat_idx = np.abs(lats - mac['lat']).argmin()
    lon_idx = np.abs(lons - mac['lon']).argmin()
    levels  = file.variables['lv_ISBL1'][:] #(37,0)
    time1  = file.variables['initial_time0_encoded'][:] #every 6 h
    time2  = file.variables['initial_time0_hours'][:]
    temp = file.variables['T_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
    t_name = file.variables['T_GDS0_ISBL'].long_name
    t_units = file.variables['T_GDS0_ISBL'].units
    file.close

    file = Dataset(path_data+ str(Y) +'/U_' +str(Y)+ '_'+str(1).zfill(2)+'.nc', 'r')
    u = file.variables['U_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
    file.close

    file = Dataset(path_data+ str(Y) +'/V_' +str(Y)+ '_'+str(1).zfill(2)+'.nc', 'r')
    v = file.variables['V_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
    file.close

    file = Dataset(path_data+ str(Y) +'/R_' +str(Y)+ '_'+str(1).zfill(2)+'.nc', 'r')
    rh = file.variables['R_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
    file.close

    file = Dataset(path_data+ str(Y) +'/Q_' +str(Y)+ '_'+str(1).zfill(2)+'.nc', 'r')
    q = file.variables['Q_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
    file.close
    print temp.shape, q.shape, 1
    print '----------------------'

    for m in range(2,3):

#******************************************************************************
#Temperature
#******************************************************************************
        file = Dataset(path_data+ str(Y) +'/T_' +str(Y)+ '_'+str(m).zfill(2)+'.nc', 'r')

        for i,variable in enumerate(file.variables):
        #    print '   '+str(i),variable
            if i == 4:
                current_variable = variable

        #print 'Variable: ', current_variable.upper()

        lats  = file.variables['g0_lat_2'][:] #(240,)
        lons  = file.variables['g0_lon_3'][:] #(121,)
        # Find the nearest latitude and longitude for Macquerie Island
        lat_idx = np.abs(lats - mac['lat']).argmin()
        lon_idx = np.abs(lons - mac['lon']).argmin()
        level  = file.variables['lv_ISBL1'][:] #(37,)
        lev_units=file.variables['lv_ISBL1'].units
        time11  = file.variables['initial_time0_encoded'][:] #every 6 h
        time1_units  = file.variables['initial_time0_encoded'].units
        time21  = file.variables['initial_time0_hours'][:]
        time2_units  = file.variables['initial_time0_hours'].units

        temp1 = file.variables['T_GDS0_ISBL'][:,:,lat_idx ,lon_idx] #(time, 37, 121, 240)
        t_name1 = file.variables['T_GDS0_ISBL'].long_name
        t_units1 = file.variables['T_GDS0_ISBL'].units
        print temp1.shape, m
        print '------------------'
        print t_name, t_units
        file.close

#******************************************************************************
#U component
#******************************************************************************
        file = Dataset(path_data+ str(Y) +'/U_' +str(Y)+ '_'+str(m).zfill(2)+'.nc', 'r')

        for i,variable in enumerate(file.variables):
        #    print '   '+str(i),variable
            if i == 4:
                current_variable = variable

        u1 = file.variables['U_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
        u_name = file.variables['U_GDS0_ISBL'].long_name
        u_units = file.variables['U_GDS0_ISBL'].units
        print u_name, u_units
        file.close
#******************************************************************************
#V component
#******************************************************************************
        file = Dataset(path_data+ str(Y) +'/V_' +str(Y)+ '_'+str(m).zfill(2)+'.nc', 'r')

        for i,variable in enumerate(file.variables):
        #    print '   '+str(i),variable
            if i == 4:
                current_variable = variable

        v1 = file.variables['V_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
        v_name = file.variables['V_GDS0_ISBL'].long_name
        v_units = file.variables['V_GDS0_ISBL'].units
        print v_name, v_units
        file.close

#******************************************************************************
#Relative Humidity
#******************************************************************************
        file = Dataset(path_data+ str(Y) +'/R_' +str(Y)+ '_'+str(m).zfill(2)+'.nc', 'r')

        for i,variable in enumerate(file.variables):
        #    print '   '+str(i),variable
            if i == 4:
                current_variable = variable

        rh1 = file.variables['R_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
        rh_name = file.variables['R_GDS0_ISBL'].long_name
        rh_units = file.variables['R_GDS0_ISBL'].units
        print rh_name, rh_units
        file.close

#******************************************************************************
#Specific Humidity
#******************************************************************************
        file = Dataset(path_data+ str(Y) +'/Q_' +str(Y)+ '_'+str(m).zfill(2)+'.nc', 'r')

        for i,variable in enumerate(file.variables):
        #    print '   '+str(i),variable
            if i == 4:
                current_variable = variable

        q1 = file.variables['Q_GDS0_ISBL'][:,:,lat_idx ,lon_idx]
        q_units = file.variables['Q_GDS0_ISBL'].units
        q_name = file.variables['Q_GDS0_ISBL'].long_name
        print q_name, q_units
        file.close


#******************************************************************************
#Concadenar
#******************************************************************************
        time1=np.concatenate((time1,time11), axis=0)
        time2=np.concatenate((time2,time21), axis=0)

        temp=np.concatenate((temp,temp1), axis=0)
        u=np.concatenate((u,u1), axis=0)
        v=np.concatenate((v,v1), axis=0)
        rh=np.concatenate((rh,rh1), axis=0)
        q=np.concatenate((q,q1), axis=0)
        print '--------------------------'
        print temp.shape, q.shape, m
        print '--------------------------'


#*****************************************************************************\******************************************************************************
#Write mat-file MAC

 #   sio.savemat(path_data_save+ 'ERAImac_'+str(Y)+'.mat', {'time1':time1,'time2':time2,'temp':temp, 't_units':t_units,'u':u, 'u_units':u_units,'v':v, 'v_units':v_units,'rh':rh, 'rh_units':rh_units,'q':q, 'q_units':q_units,'levels':levels, 'lev_units':lev_units,'time1_units':time1_units,'time2_units':time2_units})

    print Y
    print '*************************************************'
