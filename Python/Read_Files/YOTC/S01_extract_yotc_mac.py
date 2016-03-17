import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
#from glob import glob
#from scipy.interpolate import griddata

year=2009

base_dir='../../../../Data/YOTC/'

#temperatura
#******************************************************************************
file = Dataset(base_dir+str(year)+'_temp.nc', 'r')

#print ' '
#print ' '
#print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

#lats  = file.variables['latitude'][:] #(240,0)
#lons  = file.variables['longitude'][:] #(121,0)
level  = file.variables['level'][:] #(20,0)
time  = file.variables['time'][:] #(1460,)
temp = file.variables['t'][:,:,97,106] #(1460, 20, 121, 240)
t_units = file.variables['t'].units
print t_units
file.close

#wind u
#******************************************************************************
file = Dataset(base_dir+str(year)+'_u.nc', 'r')

#print ' '
#print ' '
print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

u = file.variables['u'][:,:,97,106] #(1460, 20, 121, 240)
u_units = file.variables['u'].units
print u_units
file.close
#******************************************************************************

file = Dataset(base_dir+str(year)+'_v.nc', 'r')

#print ' '
#print ' '
print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
#    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
#print ' '
print 'Variable: ', current_variable.upper()

v = file.variables['v'][:,:,97,106] #(1460, 20, 121, 240)
v_units = file.variables['v'].units
print v_units
file.close

#*****************************************************************************\******************************************************************************

#Crear fecha de inicio leyendo time
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since

#Arreglo de fechas
date_range = pd.date_range(date_ini, periods=len(time), freq='6H')

#Dataframe datos per level
nlevel=range(1,len(level)+1)
#Crear listas con variables y unirlas a dataframe 3D
t_list=temp.tolist()
u_list=u.tolist()
v_list=v.tolist()

data={'temp':t_list, 'u':u_list, 'v':v_list}
df=pd.DataFrame(data=data, index=date_range)

#df['u'].iloc[10] accede por fila y variable
#df.loc[datetime(2009,1,1,0,0)] acceder por fecha
#If index of df is sparse, it can help to slice a second dataframe just using df2.loc[df_idx_sparse]the df index:
#dIf index of df is sparse, it can help to slice a second dataframe just using the df index:
foo1=datetime(2009,1,1,0,0)
foo2=datetime(2009,2,2,0,0)

df_idx_sparse = df.loc[foo1:foo2].index
df2=df.loc[df_idx_sparse]

#df2.columns=['temp','u','v']

