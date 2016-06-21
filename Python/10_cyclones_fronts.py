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
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

latMac=-54.50;
lonMac=158.95;
#*****************************************************************************\
#*****************************************************************************\
# Reading CSV with Boundary Layer Clasification
#*****************************************************************************\
#*****************************************************************************\
df_macyotc_final= pd.read_csv(path_data + 'df_macyotc_final925.csv', sep='\t', parse_dates=['Date'])
# df_mac_final= pd.read_csv(path_data + 'df_mac_final.csv', sep='\t', parse_dates=['Date'])
df_yotc_all= pd.read_csv(path_data + 'df_yotc_all925.csv', sep='\t', parse_dates=['Date'])
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
'cyc_lat':latitud,
'cyc_lon':longitud,
'cyc_dist':dist_low}

df_cyc1 = pd.DataFrame(data=dm,index=mydates)

#*****************************************************************************\
#Closest Cyclon

df2 = df_cyc1.groupby(df_cyc1.index)
target = df2['cyc_dist'].min().values
idx=[np.where(df_cyc1['cyc_dist']==v)[0][0] for v in target]
df_cyc2 = df_cyc1.iloc[idx].copy()

# Eliminate Duplicate Soundings
df_cyc3=df_cyc2.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

#Date index del periodo 2006-2010
date_index_all = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')
df_cyc4=df_cyc3.reindex(date_index_all)
#*****************************************************************************\
#Calcular distancia from low
df_cyc4['cyc_dy']=(-df_cyc3['cyc_lat']+latMac)
df_cyc4['cyc_dx']=(-df_cyc3['cyc_lon']+lonMac)

df_cyc4.index.name = 'Fecha'
#df_mac= df_mac_final.set_index('Date')
df_yotc= df_yotc_all.set_index('Date')
#df_yotc= df_yotc.set_index('Date')
df_my= df_macyotc_final.set_index('Date')

#Unir datraframe mac con cyclones
#df_maccyc=pd.concat([df_mac, df_cyc4],axis=1)

#Unir datraframe yotc con cyclones
df_yotccyc=pd.concat([df_yotc, df_cyc4],axis=1)

#Unir datraframe mac-yotc con cyclones
df_mycyc=pd.concat([df_my, df_cyc4],axis=1)

#*****************************************************************************\
#Saving CSV
#*****************************************************************************\
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

# df_maccyc.to_csv(path_data_save + 'df_maccyc925.csv', sep='\t', encoding='utf-8')
# df_yotccyc.to_csv(path_data_save + 'df_yotccyc925.csv', sep='\t', encoding='utf-8')
# df_mycyc.to_csv(path_data_save + 'df_mycyc925.csv', sep='\t', encoding='utf-8')

#*****************************************************************************\
#*****************************************************************************\
#                           FRONTS
#*****************************************************************************\
#*****************************************************************************\

#Reading
path_front=base_dir+'/Dropbox/Monash_Uni/SO/MAC/MatFiles/files_fronts/'
#cada col es un frente, cada row es una posicion del frente
matb1= sio.loadmat(path_front+'FRONTS_2006.mat')
matb2= sio.loadmat(path_front+'FRONTS_2007.mat')
matb3= sio.loadmat(path_front+'FRONTS_2008.mat')
matb4= sio.loadmat(path_front+'FRONTS_2009.mat')
matb5= sio.loadmat(path_front+'FRONTS_2010.mat')
cf06=matb1['cold_fronts'][:] #(2,100 row ,200 col ,1460)
cf07=matb2['cold_fronts'][:]
cf08=matb3['cold_fronts'][:]
cf09=matb4['cold_fronts'][:]
cf10=matb5['cold_fronts'][:]

cold_fronts = np.concatenate((cf06,cf07,cf08,cf09,cf10), axis=3)
#cold_fronts = np.array(cf06)
#cold_fronts = np.concatenate((cf06,cf07), axis=3)
#n=7304
#n=4384
_,_,_,n=cold_fronts.shape

Clat=np.array([100,200,n])
Clon=np.array([100,200,n])

Clon=cold_fronts[1,:,:,:]
Clat=cold_fronts[0,:,:,:]

Clat[Clat==-9999]=np.nan
Clon[Clon==-9999]=np.nan
#*****************************************************************************\
#Creating Dataframe Pandas
#*****************************************************************************\
#Clatl=Clat.tolist()
#Clonl=Clon.tolist()

date_fronts = pd.date_range('2006-01-01 00:00', periods=n, freq='6H')
#dff={'cfro_lat':Clatl,'cfro_lon':Clonl }

#df_front1 = pd.DataFrame(data=dff,index=date_fronts)
#df_front1.index.name = 'Date'

##Every 12 hours
#date_index_12 = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')
#df_front=df_front1.reindex(date_index_12)

#*****************************************************************************\
#Reindex YOTC and MAC  AVE to Front Dates
df_mycyc=df_mycyc.reindex(date_fronts)
df_mycyc.index.name = 'Date'
#*****************************************************************************\

#n=20 #total time
nl=200 #200

Clon_cyc=np.empty([100,n])*np.nan
Clat_cyc=np.empty([100,n])*np.nan

DCfront=np.empty([nl,n])*np.nan
CCfront=np.empty([n])*np.nan

for ntime in range(0,n):
    x0=df_mycyc['cyc_lon'][ntime]
    y0=df_mycyc['cyc_lat'][ntime]

    for i in range(0,nl): #200

        x=Clon[:,i,ntime]
        y=Clat[:,i,ntime]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]



        if len(x)>1 and np.count_nonzero(x)>1 and np.count_nonzero(y)>1: #At least two points
            #******************************************************************
            #Line equation
            m,b=np.polyfit(x,y,1)
            recta=m*x+b

            x2=x[0]
            x1=x[-1]
            y2=recta[0]
            y1=recta[-1]
            #y2=-52
            #******************************************************************
            #Calculate Distance MAC to front line
            px = x2-x1
            py = y2-y1
            something = px*px + py*py
            u =  ((x0 - x1) * px + (y0 - y1) * py) / float(something)

            if u > 1: #Revisar para dar un rango de error fuera
                u = 1
            elif u < 0:
                u = 0

            xx = x1 + u * px
            yy = y1 + u * py

            dx = xx - x0
            dy = yy - y0
            dist = math.sqrt(dx*dx + dy*dy)

            #******************************************************************
        else:
            dist=np.nan

    #Construir nueva matriz (1,200,1460)
        DCfront[i,ntime]=np.array(dist)

#*****************************************************************************\


#Calculate closest front to Low
for ntime in range(0,n):
    if np.all(np.isnan(DCfront[:,ntime])): #read column if all are nan
        CCfront[ntime] = np.nan
        Clon_cyc[:,ntime]=np.nan
        Clat_cyc[:,ntime]=np.nan
    else:
        idx = np.nanargmin(np.abs(DCfront[:,ntime])) #position closest to zero
        CCfront[ntime] = DCfront[idx,ntime]
        Clon_cyc[:,ntime]=Clon[:,idx,ntime]
        Clat_cyc[:,ntime]=Clat[:,idx,ntime]
#Within 15 degrees
for ntime in range(0,n):
    if (CCfront[ntime])>15:
        Clon_cyc[:,ntime]=np.nan
        Clat_cyc[:,ntime]=np.nan
        CCfront[ntime] = np.nan
    else:
        Clon_cyc[:,ntime]=Clon_cyc[:,ntime]
        Clat_cyc[:,ntime]=Clat_cyc[:,ntime]
        CCfront[ntime] = CCfront[ntime]

#*****************************************************************************\
from pylab import plot,show, grid

# nc=6
# plot(df_mycyc['cyc_lon'][nc],df_mycyc['cyc_lat'][nc],'ko',
#     Clon_cyc[:,nc],Clat_cyc[:,nc],'bo')
# grid()
# show()
#*****************************************************************************\
#Dataframe Fronts-Cyclones
#*****************************************************************************\
date_fronts2 = pd.date_range('2006-01-01 00:00', periods=n, freq='6H')

Clat_list=Clat_cyc.T.tolist()
Clon_list=Clon_cyc.T.tolist()

#dff={'cfro_lat':Clat_list,'cfro_lon':Clon_list, 'cycfro_dist':CCfront}
dff={'cycfro_dist':CCfront}

df_fro1 = pd.DataFrame(data=dff,index=date_fronts2)
df_fro1.index.name = 'Date'

#*****************************************************************************\
#Every 12 hours
date_index_12 = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')
df_fro2=df_fro1.reindex(date_index_12)

#MAC AVE
df_mycyc=df_mycyc.reindex(date_index_12)
df_mycyc.index.name = 'Date'
df_mycycfro=pd.concat([df_mycyc, df_fro2],axis=1)

#YOTC
df_yotccycfro=pd.concat([df_yotccyc, df_fro2],axis=1)

#*****************************************************************************\
#Unir datraframe mac-yotc con cyclones


# df_mycycfro.to_csv(path_data_save + 'df_mycycfro.csv', sep=',', encoding='utf-8')
# df_mycycfro.to_csv(path_data_save + 'df_mycycfro2.csv', sep='\t', encoding='utf-8')
#*****************************************************************************\


#*****************************************************************************\
#*****************************************************************************\
#                           MAPS Variables
#*****************************************************************************\
#*****************************************************************************\
df_cycfro=df_mycycfro[np.isfinite(df_mycycfro['cycfro_dist'])]
path_data_savefig=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/cyc_front/'
#*****************************************************************************\
Lat1=np.array(df_cycfro['cyc_dy'])
Lon1=np.array(df_cycfro['cyc_dx'])
Data1 = np.array(df_cycfro['T 925'])
Data2 = np.array(df_cycfro['Mix R 925'])
Data3 = np.array(df_cycfro['RH 925'])
Data4=np.sqrt(np.array(df_cycfro['u 925'])**2 + np.array(df_cycfro['u 925'])**2) #wind speed

#*****************************************************************************\
# Temperature
#*****************************************************************************\
zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi = zi / counts
zi = np.ma.masked_invalid(zi)

#*****************************************************************************\
#Water Content
#*****************************************************************************\
zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data2, normed=False)
counts1, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)

#*****************************************************************************\
#Water Content
#*****************************************************************************\

zi2, yi2, xi2 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data3, normed=False)
counts2, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi2 = zi2 / counts2
zi2 = np.ma.masked_invalid(zi2)

#*****************************************************************************\
#Wind Speed
#*****************************************************************************\

zi3, yi3, xi3 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data4, normed=False)
counts3, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi3 = zi3 / counts3
zi3 = np.ma.masked_invalid(zi3)


#*****************************************************************************\
row = 1
column = 4
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(22,5))
ax0, ax1, ax2, ax3 = axes.flat
cmap='seismic'
#cmap='cool'
#*****************************************************************************\
vmax=290
vmin=260
limu=vmax+5
v = np.arange(vmin, limu, 5)

img1= ax0.pcolor(xi, yi, zi, cmap=cmap,vmin=vmin, vmax=vmax)
div = make_axes_locatable(ax0)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img1, cax=cax, format="%.0f")
#cbar.ax.set_title('     height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax0.set_title('Temperature (K) 925 hPa', size=18)
ax0.set_yticks([0], minor=True)
ax0.set_xticks([0], minor=True)
ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax0.grid(b=True, which='major', color='grey', linestyle='--')
ax0.set_ylabel('Distance from low (deg)', size=18)
ax0.set_xlabel('Distance from low (deg)', size=18)
ax0.margins(0.05)
#*****************************************************************************\
vmax=10
vmin=0
limu=vmax+5
v = np.arange(vmin, limu, 1)

img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=vmin, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('Mixing Ratio (g kg$^{-1}$) 925 hPa', size=18)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img2, cax=cax, format="%.0f",ticks=v)
#cbar.ax.set_title('     height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax1.set_xlabel('Distance from low (deg)', size=18)
ax1.margins(0.05)
#*****************************************************************************\
vmax=100
vmin=50
limu=vmax+5
v = np.arange(vmin, limu, 5)

img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=vmin, vmax=vmax)
ax2.set_title('Relative Humidity (%) 925 hPa', size=18)
div = make_axes_locatable(ax2)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax, format="%.0f",ticks=v)
#cbar.set_label(' height (mts.)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax2.set_xlabel('Distance from low (deg)', size=18)
ax2.margins(0.05)
plt.tight_layout()

#*****************************************************************************\
vmax=60
vmin=0
limu=vmax+5
v = np.arange(vmin, limu, 5)

img4= ax3.pcolor(xi3, yi3, zi3, cmap=cmap,vmin=vmin, vmax=vmax)
ax3.set_title('Wind Speed (m s$^{-1}$) 925 hPa', size=18)
div = make_axes_locatable(ax3)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img4, cax=cax, format="%.0f",ticks=v)
#cbar.set_label(' height (mts.)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax3.set_yticks([0], minor=True)
ax3.set_xticks([0], minor=True)
ax3.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax3.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax3.set_xlabel('Distance from low (deg)', size=18)
ax3.margins(0.05)
plt.tight_layout()

fig.savefig(path_data_savefig + 'variables_my.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#                           MAPS Height-Streng
#*****************************************************************************\
#*****************************************************************************\
Lat1=np.array(df_cycfro['cyc_dy'])
Lon1=np.array(df_cycfro['cyc_dx'])
Data1 = np.array(df_cycfro['1ra Inv'])
Data2 = np.array(df_cycfro['2da Inv'])
Data3 = np.array(df_cycfro['Strg 1inv'])
Data4 = np.array(df_cycfro['Strg 2inv'])

#*****************************************************************************\
zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi = zi / counts
zi = np.ma.masked_invalid(zi)
#*****************************************************************************\
zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data2, normed=False)
counts1, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
zi2, yi2, xi2 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data3, normed=False)
counts2, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi2 = zi2 / counts2
zi2 = np.ma.masked_invalid(zi2)
#*****************************************************************************\
zi3, yi3, xi3 = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data4, normed=False)
counts3, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi3 = zi3 / counts3
zi3 = np.ma.masked_invalid(zi3)
#*****************************************************************************\
row = 1
column = 4
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(22,5))
ax0, ax1, ax2, ax3 = axes.flat
#cmap='seismic'
#*****************************************************************************\
vmin=0
vmax=2400
limu=vmax+300
v = np.arange(vmin, limu, 300)
cmap='Blues'

img1= ax0.pcolor(xi, yi, zi, cmap=cmap,vmin=vmin, vmax=vmax)
div = make_axes_locatable(ax0)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img1, cax=cax, format="%.0f")
#cbar.ax.set_title('     height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax0.set_title('Height Main Inversion (mts)', size=18)
ax0.set_yticks([0], minor=True)
ax0.set_xticks([0], minor=True)
ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax0.grid(b=True, which='major', color='grey', linestyle='--')
ax0.set_ylabel('Distance from low (deg)', size=18)
ax0.set_xlabel('Distance from low (deg)', size=18)
ax0.margins(0.05)
#*****************************************************************************\
cmap='Reds'

img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=vmin, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('Height Secondary Inversion (mts)', size=18)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img2, cax=cax, format="%.0f",ticks=v)
#cbar.ax.set_title('     height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax1.set_xlabel('Distance from low (deg)', size=18)
ax1.margins(0.05)
#*****************************************************************************\
vmax=0.1
vmin=0
limu=vmax+0.01
v = np.arange(vmin, limu, 0.01)
cmap='Blues'

img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=vmin, vmax=vmax)
ax2.set_title('Strength Main Inversion (K m$^{-1}$ )', size=18)
div = make_axes_locatable(ax2)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax,ticks=v)
#cbar.set_label(' height (mts.)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax2.set_xlabel('Distance from low (deg)', size=18)
ax2.margins(0.05)
plt.tight_layout()

#*****************************************************************************\
cmap='Reds'

img4= ax3.pcolor(xi3, yi3, zi3, cmap=cmap,vmin=vmin, vmax=vmax)
ax3.set_title('Strength Secondary Inversion (K m$^{-1}$ )', size=18)
div = make_axes_locatable(ax3)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img4, cax=cax,ticks=v)
#cbar.set_label(' height (mts.)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax3.set_yticks([0], minor=True)
ax3.set_xticks([0], minor=True)
ax3.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax3.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax3.set_xlabel('Distance from low (deg)', size=18)
ax3.margins(0.05)
plt.tight_layout()

fig.savefig(path_data_savefig + 'heightstrenght_my.eps', format='eps', dpi=1200)



df1=df_yotccycfro[np.isfinite(df_yotccycfro['Clas'])]

