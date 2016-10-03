import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime
import os
import csv
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'

latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
# Reading CSV
#*****************************************************************************\
df_macyotc_final= pd.read_csv(path_data + 'df_macyotc_20062010.csv', sep='\t', parse_dates=['Date'])
df_mac_final= pd.read_csv(path_data + 'df_mac_20062010.csv', sep='\t', parse_dates=['Date'])
df_yotc_all= pd.read_csv(path_data + 'df_yotc_20082010.csv', sep='\t', parse_dates=['Date'])
df_yotc= pd.read_csv(path_data + 'df_yotc_20062010.csv', sep='\t', parse_dates=['Date'])
#*****************************************************************************\
#*****************************************************************************\
#                           CYCLONES
#*****************************************************************************\
#*****************************************************************************\
#Reading
path_mcms=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MCMS/'
#Reading Multiples Files
fnames = glob(path_mcms+'*.txt')
arrays = [np.loadtxt(f, delimiter=' ') for f in fnames]
array_cyc = np.concatenate(arrays, axis=0)
#np.savetxt('result.txt', final_array, fmt='%.2f')
#*****************************************************************************\
#Filtrar
#*****************************************************************************\
#Latitud
CoLat=array_cyc[:,5]
lat = 90.0 - (CoLat*0.01);
indlan = np.nonzero(lat<-30.)
lat2=lat[indlan] #Eliminate North Hemisphere

cycf1=array_cyc[indlan]

#Lontitud
CoLon=cycf1[:,6]
lon=CoLon*0.01;
indlon = np.nonzero(lon<180)
lon2=lon[indlon] #Eliminate

cyc_close=cycf1[indlon]


#Hora
hora=cyc_close[:,3]
indh1 = np.nonzero(hora!=6)
cyc_h1=cyc_close[indh1]
hora1=cyc_h1[:,3]
indh2 = np.nonzero(hora1!=18)
cyc_filt=cyc_h1[indh2]


#*****************************************************************************\
#Latitud
lati = 90.0 - (cyc_filt[:,5]*0.01);
#Longitud
longi=cyc_filt[:,6]*0.01;
#*****************************************************************************\
#Distance from MAC
dy=(-lati+latMac);
dx=(-longi+lonMac);

#15 degrees from MAC
lim=15;

distance=np.sqrt(dy**2+dx**2);
ind_dist = np.nonzero(distance<=lim)

latitud=lati[ind_dist]
longitud=longi[ind_dist]
cyc_mac=cyc_filt[ind_dist]
dist_low=distance[ind_dist]
del dy,dx
#*****************************************************************************\
#Creating datetime variables
array=cyc_mac[:,0:4] #Array fechas

yy=array[:,0].astype(int)
mm=array[:,1].astype(int)
dd=array[:,2].astype(int)
hh=array[:,3].astype(int)

ndates,_=array.shape

mydates=np.array([])
for n in range(0,ndates):
    mydates=np.append(mydates, datetime(yy[n], mm[n], dd[n],hh[n],0,0))

#*****************************************************************************\
#Dataframe Pandas

dm={'USI':cyc_mac[:,-1],
'lat':latitud,
'lon':longitud,
'dist':dist_low}

df_cyc1 = pd.DataFrame(data=dm,index=mydates)

#*****************************************************************************\
#Closest Cyclon

df2 = df_cyc1.groupby(df_cyc1.index)
target = df2['dist'].min().values
idx=[np.where(df_cyc1['dist']==v)[0][0] for v in target]
df_cyc2 = df_cyc1.iloc[idx].copy()

# Eliminate Duplicate Soundings
df_cyc3=df_cyc2.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

#Date index del periodo 2006-2010
date_index_all = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')
df_cyc4=df_cyc3.reindex(date_index_all)
#*****************************************************************************\
#Calcular distancia from low
df_cyc4['dy']=(-df_cyc3['lat']+latMac)
df_cyc4['dx']=(-df_cyc3['lon']+lonMac)

df_cyc4.index.name = 'Fecha'
df_mac= df_mac_final.set_index('Date')
df_yotc= df_yotc_all.set_index('Date')
#df_yotc= df_yotc.set_index('Date')
df_my= df_macyotc_final.set_index('Date')

#Unir datraframe mac con cyclones
df_maccyc=pd.concat([df_mac, df_cyc4],axis=1)

#Unir datraframe yotc con cyclones
df_yotccyc=pd.concat([df_yotc, df_cyc4],axis=1)

#Unir datraframe mac-yotc con cyclones
df_mycyc=pd.concat([df_my, df_cyc4],axis=1)

#*****************************************************************************\
#Saving CSV
#*****************************************************************************\

df_maccyc.to_csv(path_data + 'df_maccyc.csv', sep='\t', encoding='utf-8')
df_yotccyc.to_csv(path_data + 'df_yotccyc.csv', sep='\t', encoding='utf-8')
df_mycyc.to_csv(path_data + 'df_mycyc.csv', sep='\t', encoding='utf-8')


