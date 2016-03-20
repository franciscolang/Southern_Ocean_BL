import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
import pandas as pd
import scipy.io as sio
import os
import csv
#import skewt import SkewT

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monas Uni/SO/MAC/Data'

#*****************************************************************************\

mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}
lat_mac = mac['lat']
lon_mac = mac['lon']

#*****************************************************************************\
mat= sio.loadmat(path_data+'/YOTC/mat/yotc_mac2008-2010.mat')
#*****************************************************************************\
#Crear fecha de inicio leyendo time
time= mat['time'][0]
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Arreglo de fechas
date_range = pd.date_range(date_ini, periods=len(time), freq='12H')
#*****************************************************************************\
#Reading variables
temp= mat['temp'][:] #K
u= mat['u'][:]
v= mat['u'][:]
q= mat['q'][:] #kg/kg
#*****************************************************************************\
#Leyendo Alturas y press
file_levels = np.genfromtxt('levels.csv', delimiter=',')

hlev_yotc=file_levels[:30,6]
plev_yotc=file_levels[:30,3]

#*****************************************************************************\
#Calculate Virtual Temperature

theta=(temp)*(1000/plev_yotc)**287;
theta0=(temp[0,-1])*(1000/plev_yotc[-1]);
thetav=(1+0.61*(1/1000))*theta;


#*****************************************************************************\
#Dataframe datos per level
nlevel=range(1,31)
#Crear listas con variables y unirlas a dataframe 3D
t_list=temp.tolist()
u_list=u.tolist()
v_list=v.tolist()
q_list=q.tolist()

data={'temp':t_list, 'q':q_list,'u':u_list, 'v':v_list}
df=pd.DataFrame(data=data, index=date_range)
#*****************************************************************************\
