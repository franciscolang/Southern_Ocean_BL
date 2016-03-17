import numpy as np
#import scipy.io as sio
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
#from glob import glob
#from scipy.interpolate import griddata

year=2009

#temperatura
#******************************************************************************
file = Dataset('data_files/'+str(year)+'_temp.nc', 'r')

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
file = Dataset('data_files/'+str(year)+'_u.nc', 'r')

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
#******************************************************************************

#Crear fecha de inicio leyendo time
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since

#Arreglo de fechas
date_range = pd.date_range(date_ini, periods=len(time), freq='6H')

cols=[u[:,0], temp[:,0]]
var=np.transpose(cols)

f_t=pd.DataFrame(data=var,index=date_range)

#Dataframe datos per level
nlevel=range(1,len(level)+1)
df_t=pd.DataFrame(data=temp,index=date_range)
df_u=pd.DataFrame(data=u,index=date_range)

#df_t.to_csv('./prueba_csv', date_format='%Y-%m-%d %H:%M:%S', float_format='%9.6f', header=None, sep=' ')

df_lv1=df_t.ix[:, 0]

#******************************************************************************
#3D
u1=u[0,:]
u2=u[1,:]
t1=temp[0,:]
t2=temp[1,:]
date_range1 = pd.date_range('2015-01-01 00:00', periods=2, freq='1T')

cols=['t','u']
 
nandf = pd.DataFrame(index=date_range1, columns=cols)
nandf['u']=[u1,u2]
nandf['t']=[t1,t2]


#********************************	
timestamp=date_range

t_list=temp.tolist()
u_list=u.tolist()

data={'temp':t_list, 'u':u_list}	
df=pd.DataFrame(data=data, index=date_range)