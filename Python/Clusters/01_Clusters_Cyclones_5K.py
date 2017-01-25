import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import plot,show, grid, legend
from matplotlib.pyplot import rcParams,figure,show,draw
from numpy import inf
from scipy import array, arange, exp
from glob import glob
from skewt import SkewT

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/'
#path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/fronts/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/clusters/Cyclones/'


latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\
#*****************************************************************************\
#925-850-700

df_cluster= pd.read_csv(path_data + 'Soundings_MAC_Output_ClusterCyc.csv', sep=',', parse_dates=['Date'])

clus=np.array(df_cluster['Cluster 5'])
dist_clus=np.array(df_cluster['Distance 5'])

#*****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
# ****************************************************************************\

path_data_erai=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/'
matb1= sio.loadmat(path_data_erai+'ERAImac_1995.mat')

#Pressure Levels
pres_erai=matb1['levels'][:] #hPa
pres_ei=pres_erai[0,::-1]

# ****************************************************************************\

path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
matb1= sio.loadmat(path_databom+'BOM_2006.mat')
bom_in=matb1['BOM_S'][:]
timesd= matb1['time'][:]
bom=bom_in

for y in range(2007,2011):
    matb= sio.loadmat(path_databom+'BOM_'+str(y)+'.mat')
    bom_r=matb['BOM_S'][:]
    timesd_r= matb['time'][:]
    bom=np.concatenate((bom,bom_r), axis=2)
    timesd=np.concatenate((timesd,timesd_r), axis=1)

ni=bom.shape
#*****************************************************************************\
#Delete special cases
indexdel1=[]
indexdel2=[]
#Delete when array is only NaN
for j in range(0,ni[2]):
    if np.isnan(np.nansum(bom[:,0,j]))==True:
        indexdel1=np.append(indexdel1,j)

bom=np.delete(bom, indexdel1,axis=2)
ni=bom.shape
timesd=np.delete(timesd, indexdel1)

#Delete when max height is lower than 2500 mts
for j in range(0,ni[2]):
    if np.nanmax(bom[:,1,j])<2500:
        indexdel2=np.append(indexdel2,j)

bom=np.delete(bom, indexdel2,axis=2) #(3000,8,3545)
ni=bom.shape
timesd=np.delete(timesd, indexdel2)
#*****************************************************************************\

#Convert Matlab datenum into Python datetime source: https://gist.github.com/vicow)
def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=int(days)) \
        + timedelta(hours=int(hours)) \
        + timedelta(minutes=int(minutes)) \
        + timedelta(seconds=round(seconds)) \
        - timedelta(days=366)


#*****************************************************************************\
#Separation of Variables
pres=bom[:,0,:].reshape(ni[0],ni[2])
hght=bom[:,1,:].reshape(ni[0],ni[2])
temp=bom[:,2,:].reshape(ni[0],ni[2]) #^oC
dewp=bom[:,3,:].reshape(ni[0],ni[2]) #^oC
mixr=bom[:,5,:].reshape(ni[0],ni[2]) #g/kg
wdir=bom[:,6,:].reshape(ni[0],ni[2])
wspd=bom[:,7,:].reshape(ni[0],ni[2])
relh=bom[:,4,:].reshape(ni[0],ni[2])

u=wspd*(np.cos(np.radians(270-wdir)))
v=wspd*(np.sin(np.radians(270-wdir)))

#Elimanting soundings with all nans
pres=pres.T

mask = np.all(np.isnan(pres), axis=1)
hght=hght.T[~mask].T
temp=temp.T[~mask].T
dewp=dewp.T[~mask].T
mixr=mixr.T[~mask].T
wdir=wdir.T[~mask].T
wspd=wspd.T[~mask].T
relh=relh.T[~mask].T
timesd=timesd[~mask]
u=u.T[~mask].T
v=v.T[~mask].T
pres=pres[~mask].T


q=np.empty(mixr.shape)*np.nan
for i in range(0, len(timesd)):
    for j in range(0,3000):
        q[j,i]=(float(mixr[j,i])/1000.)/(1+(float(mixr[j,i])/1000.))

ni=pres.shape
#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data ERA-i Levels
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Extrapolation function
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

#*****************************************************************************\
#Definition Variables Inp.
temp_pres=np.zeros((len(pres_ei),ni[1]),'float')
mixr_pres=np.zeros((len(pres_ei),ni[1]),'float')
u_pres=np.zeros((len(pres_ei),ni[1]),'float')
v_pres=np.zeros((len(pres_ei),ni[1]),'float')
relh_pres=np.zeros((len(pres_ei),ni[1]),'float')
q_pres=np.zeros((len(pres_ei),ni[1]),'float')
dewp_pres=np.zeros((len(pres_ei),ni[1]),'float')

#Linear Interpolation
for j in range(0,ni[1]):
    x=np.log(pres[~np.isnan(temp[:,j]),j])

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yu=u[~np.isnan(u[:,j]),j]
    yv=v[~np.isnan(v[:,j]),j]
    yr=relh[~np.isnan(relh[:,j]),j]
    yq=q[~np.isnan(q[:,j]),j]
    yd=dewp[~np.isnan(dewp[:,j]),j]


    f_iq = interp1d(x, yq)
    f_it = interp1d(x, yt)
    f_im = interp1d(x, ym)
    f_iu = interp1d(x, yu)
    f_iv = interp1d(x, yv)
    f_ir = interp1d(x, yr)
    f_id = interp1d(x, yd)

    #f_xq = extrap1d(f_iq)

    #Output Variables Interpolation
    q_pres[:,j]=extrap1d(f_iq)(np.log(pres_ei))
    temp_pres[:,j]=extrap1d(f_it)(np.log(pres_ei))
    mixr_pres[:,j]=extrap1d(f_im)(np.log(pres_ei))
    u_pres[:,j]=extrap1d(f_iu)(np.log(pres_ei))
    v_pres[:,j]=extrap1d(f_iv)(np.log(pres_ei))
    relh_pres[:,j]=extrap1d(f_ir)(np.log(pres_ei))
    dewp_pres[:,j]=extrap1d(f_id)(np.log(pres_ei))

#*****************************************************************************\
# Wind Speed and Direction
wspd_pres=np.sqrt(u_pres**2 + v_pres**2)
wdir_pres=np.arctan2(-u_pres, -v_pres)*(180/np.pi)
wdir_pres[(u_pres == 0) & (v_pres == 0)]=0


#*****************************************************************************\
#Initialization Variables

spec_hum_my=q_pres

#Height Calculation
#*****************************************************************************\
Rd=287.04 #J/(kg K)
g=9.8 #m/s2
hght_my_exp=np.empty(temp_pres.shape)
hght_my_exp[:]=np.nan

for i in range(0,ni[1]):
    for j in range(0,len(pres_ei)):
        hght_my_exp[j,i]=(Rd*(temp_pres[j,i]+273.16))/float(g)*np.log(1000/float(pres_ei[j]))

#*****************************************************************************\
temp_pres[temp_pres== inf] = np.nan
temp_pres[temp_pres== -inf] = np.nan
relh_pres[relh_pres== inf] = np.nan
relh_pres[relh_pres== -inf] = np.nan
q_pres[q_pres== inf] = np.nan
q_pres[q_pres== -inf] = np.nan

dewp_pres[dewp_pres== inf] = np.nan
dewp_pres[dewp_pres== -inf] = np.nan
u_pres[u_pres== inf] = np.nan
u_pres[u_pres== -inf] = np.nan
v_pres[v_pres== inf] = np.nan
v_pres[v_pres== -inf] = np.nan

# Wind Speed and Direction
wspd_pres=np.sqrt(u_pres**2 + v_pres**2)
wdir_pres=np.arctan2(-u_pres, -v_pres)*(180/np.pi)
wdir_pres[(u_pres == 0) & (v_pres == 0)]=0

relhum_my=relh_pres.T
temp_my=temp_pres.T
u_my=u_pres.T
v_my=v_pres.T
mixr_my=mixr_pres.T
q_my=q_pres.T*1000
dewp_my=dewp_pres.T
hght_my=hght_my_exp.T
wsp_my=wspd_pres.T
wdir_my=wdir_pres.T
#*****************************************************************************\
#Cambiar fechas
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
time_my_ori = np.array(timestamp)

for i in range(0,ni[1]):
    #Cuando cae 23 horas del 31 de diciembre agrega un anio
    if (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==12:
        y1=time_my[i].year
        time_my[i]=time_my[i].replace(year=y1+1,hour=0, month=1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==1:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==3:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==5:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==7:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==8:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==10:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==4:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==6:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==9:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==11:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==28 and time_my[i].month==2:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    #Bisiesto 2008
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2008:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2004:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2000:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1996:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1992:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)


    #Cuando cae 23 horas, mueve hora a las 00 del dia siguiente
    if time_my[i].hour==23 or time_my[i].hour==22:
        d1=time_my[i].day
        time_my[i]=time_my[i].replace(hour=0,day=d1+1)
    else:
        time_my[i]=time_my[i]
    #Cuando cae 11 horas, mueve hora a las 12 del mismo dia
    if time_my[i].hour==11 or time_my[i].hour==10 or time_my[i].hour==13 or time_my[i].hour==14:
        time_my[i]=time_my[i].replace(hour=12)
    else:
        time_my[i]=time_my[i]

    #Cuando cae 1 horas, mueve hora a las 0 del mismo dia
    if time_my[i].hour==1 or time_my[i].hour==2:
        time_my[i]=time_my[i].replace(hour=0)
    else:
        time_my[i]=time_my[i]

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
date_index_all = pd.date_range('2006-01-01 00:00', periods=3651, freq='12H')
#*****************************************************************************\
#CL3=np.empty(len(df_cluster))*np.nan
#*****************************************************************************\
#Dataframe MAC YOTC levels
hght_list=hght_my.tolist()
t_list=temp_my.tolist()
u_list=u_my.tolist()
v_list=v_my.tolist()
rh_list=relhum_my.tolist()
mr_list=mixr_my.tolist()
dewp_list=dewp_my.tolist()
wsp_list=wsp_my.tolist()
wdir_list=wdir_my.tolist()

dmy={'temp':t_list,
'u':u_list,
'v':v_list,
'RH':rh_list,
'dewp':dewp_list,
'mixr':mr_list,
'wsp':wsp_list,
'wdir':wdir_list,
'cluster': clus,
'dist_clus': dist_clus,
'height':hght_list}



df_clumac= pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
df_clumac = df_clumac[~df_clumac.index.duplicated(keep='first')]

df_clumac=df_clumac.reindex(date_index_all)
df_clumac.index.name = 'Date'


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
df_cyc3 = df_cyc2[~df_cyc2.index.duplicated(keep='first')]


#Date index del periodo 2006-2010
#date_index_all = pd.date_range('2008-05-01 00:00', periods=1460, freq='12H')
date_index_all = pd.date_range('2006-01-01 00:00', periods=3651, freq='12H')
df_cyc4=df_cyc3.reindex(date_index_all)
#*****************************************************************************\
#Calcular distancia from low
df_cyc4['dy']=(-df_cyc3['lat']+latMac)
df_cyc4['dx']=(-df_cyc3['lon']+lonMac)

df_cyc4.index.name = 'Date'

#Unir datraframe mac con cyclones
df_macclucyc=pd.concat([df_clumac, df_cyc4],axis=1)


df_macclucyc = df_macclucyc[np.isfinite(df_macclucyc['dist_clus'])]
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# 5 Clusters
#*****************************************************************************\
df_1all = df_macclucyc[df_macclucyc['cluster']==1]
df_2all = df_macclucyc[df_macclucyc['cluster']==2]
df_3all = df_macclucyc[df_macclucyc['cluster']==3]
df_4all = df_macclucyc[df_macclucyc['cluster']==4]
df_5all = df_macclucyc[df_macclucyc['cluster']==5]

df_1 = df_1all[np.isfinite(df_1all['dy'])]
df_2 = df_2all[np.isfinite(df_2all['dy'])]
df_3 = df_3all[np.isfinite(df_3all['dy'])]
df_4 = df_4all[np.isfinite(df_4all['dy'])]
df_5 = df_5all[np.isfinite(df_5all['dy'])]


pt1=len(df_1all)/float(len(df_macclucyc))
pt1=round(pt1,2)
pc1=len(df_1)/float(len(df_1all))
pc1=round(pc1,2)

pt2=len(df_2all)/float(len(df_macclucyc))
pt2=round(pt2,2)
pc2=len(df_2)/float(len(df_2all))
pc2=round(pc2,2)

pt3=len(df_3all)/float(len(df_macclucyc))
pt3=round(pt3,2)
pc3=len(df_3)/float(len(df_3all))
pc3=round(pc3,2)

pt4=len(df_4all)/float(len(df_macclucyc))
pt4=round(pt4,2)
pc4=len(df_4)/float(len(df_4all))
pc4=round(pc4,2)

pt5=len(df_5all)/float(len(df_macclucyc))
pt5=round(pt5,2)
pc5=len(df_5)/float(len(df_5all))
pc5=round(pc5,2)

#*****************************************************************************\
row = 2
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(19.5,13))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat
cmap='Blues'
vmax=0.03
limu=vmax+0.005
v = np.arange(0, limu, 0.005)
s_bin=12
#*****************************************************************************\
# C1
#*****************************************************************************\
Lat1=np.array(df_1['dy'])
Lon1=np.array(df_1['dx'])
Data1 = np.ones(len(df_1))

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
zi1 = zi1 / float(len(Data1))
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
v = np.arange(0, limu, 0.005)

img1= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img1, cax=cax, format="%.3f",ticks=v)
#cbar.set_label('freq.', size=12)

ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
ax1.set_ylabel('Distance from low (deg)', size=14)
ax1.margins(0.05)
ax1.set_title('Cluster 1 (P$_t$=' + str(pt1)+', P$_c$='+str(pc1) +')' ,fontsize = 14 ,fontweight='bold')

del Lat1, Lon1, Data1, zi1, yi1, xi1
#*****************************************************************************\
#C2
#*****************************************************************************\
Lat1=np.array(df_2['dy'])
Lon1=np.array(df_2['dx'])
Data1 = np.ones(len(df_2))

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
zi1 = zi1 / float(len(Data1))
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
img2= ax2.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax2)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img2, cax=cax, format="%.3f",ticks=v)

ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
ax2.margins(0.05)
ax2.set_title('Cluster 2 (P$_t$=' + str(pt2)+', P$_c$='+str(pc2) +')' ,fontsize = 14 ,fontweight='bold')

del Lat1, Lon1, Data1, zi1, yi1, xi1

#*****************************************************************************\
#C3
#*****************************************************************************\
Lat1=np.array(df_3['dy'])
Lon1=np.array(df_3['dx'])
Data1 = np.ones(len(df_3))

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
zi1 = zi1 / float(len(Data1))
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
img3= ax3.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax3)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax, format="%.3f",ticks=v)

ax3.set_yticks([0], minor=True)
ax3.set_xticks([0], minor=True)
ax3.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax3.grid(b=True, which='major', color='grey', linestyle='--')
ax3.set_xlabel('Distance from low (deg)', size=14)
ax3.set_ylabel('Distance from low (deg)', size=14)
ax3.margins(0.05)
ax3.set_title('Cluster 3 (P$_t$=' + str(pt3)+', P$_c$='+str(pc3) +')' ,fontsize = 14 ,fontweight='bold')

del Lat1, Lon1, Data1, zi1, yi1, xi1


#*****************************************************************************\
#C4
#*****************************************************************************\
Lat1=np.array(df_4['dy'])
Lon1=np.array(df_4['dx'])
Data1 = np.ones(len(df_4))

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
zi1 = zi1 / float(len(Data1))
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
img4= ax4.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax4)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img4, cax=cax, format="%.3f",ticks=v)

ax4.set_yticks([0], minor=True)
ax4.set_xticks([0], minor=True)
ax4.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax4.grid(b=True, which='major', color='grey', linestyle='--')
ax4.set_xlabel('Distance from low (deg)', size=14)
ax4.margins(0.05)
ax4.set_title('Cluster 4 (P$_t$=' + str(pt4)+', P$_c$='+str(pc4) +')' ,fontsize = 14 ,fontweight='bold')

del Lat1, Lon1, Data1, zi1, yi1, xi1

#*****************************************************************************\
#C5
#*****************************************************************************\
Lat1=np.array(df_5['dy'])
Lon1=np.array(df_5['dx'])
Data1 = np.ones(len(df_5))

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
zi1 = zi1 / float(len(Data1))
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
img5= ax5.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax5)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img5, cax=cax, format="%.3f",ticks=v)

ax5.set_yticks([0], minor=True)
ax5.set_xticks([0], minor=True)
ax5.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax5.grid(b=True, which='major', color='grey', linestyle='--')
ax5.set_xlabel('Distance from low (deg)', size=14)
ax5.margins(0.05)
ax5.set_title('Cluster 5 (P$_t$=' + str(pt5)+', P$_c$='+str(pc5) +')' ,fontsize = 14 ,fontweight='bold')

del Lat1, Lon1, Data1, zi1, yi1, xi1
fig.tight_layout()
plt.savefig(path_data_save + '5K.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              Means Profiles 4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/clusters/Cyclones/5K/'

df_CL4_G1 = df_1
df_CL4_G2 = df_2
df_CL4_G3 = df_3
df_CL4_G4 = df_4
df_CL4_G5 = df_5

#*****************************************************************************\
#C1
#*****************************************************************************\
df=df_CL4_G1
RH=np.empty([len(df),37])*np.nan
U=np.empty([len(df),37])*np.nan
V=np.empty([len(df),37])*np.nan
T=np.empty([len(df),37])*np.nan
DP=np.empty([len(df),37])*np.nan
MX=np.empty([len(df),37])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G1=np.nanmean(H,axis=0)
rhum_CL4_G1=np.nanmean(RH,axis=0)
temp_CL4_G1=np.nanmean(T,axis=0)
dewp_CL4_G1=np.nanmean(DP,axis=0)
mixr_CL4_G1=np.nanmean(MX,axis=0)
v_CL4_G1=np.nanmean(V,axis=0)
u_CL4_G1=np.nanmean(U,axis=0)
wsp_CL4_G1=np.sqrt(u_CL4_G1**2 + v_CL4_G1**2)
wdir_CL4_G1=np.arctan2(-u_CL4_G1, -v_CL4_G1)*(180/np.pi)
wdir_CL4_G1[(u_CL4_G1 == 0) & (v_CL4_G1 == 0)]=0

dewp_CL4_G1[0]=np.nan

#*****************************************************************************\
#C2
#*****************************************************************************\
df=df_CL4_G2
RH=np.empty([len(df),37])*np.nan
U=np.empty([len(df),37])*np.nan
V=np.empty([len(df),37])*np.nan
T=np.empty([len(df),37])*np.nan
DP=np.empty([len(df),37])*np.nan
MX=np.empty([len(df),37])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G2=np.nanmean(H,axis=0)
rhum_CL4_G2=np.nanmean(RH,axis=0)
temp_CL4_G2=np.nanmean(T,axis=0)
dewp_CL4_G2=np.nanmean(DP,axis=0)
mixr_CL4_G2=np.nanmean(MX,axis=0)
v_CL4_G2=np.nanmean(V,axis=0)
u_CL4_G2=np.nanmean(U,axis=0)
wsp_CL4_G2=np.sqrt(u_CL4_G2**2 + v_CL4_G2**2)
wdir_CL4_G2=np.arctan2(u_CL4_G2, v_CL4_G2)*(180/np.pi)+180
wdir_CL4_G2[(u_CL4_G2 == 0) & (v_CL4_G2 == 0)]=0

dewp_CL4_G2[0]=np.nan

#*****************************************************************************\
#C3
#*****************************************************************************\
df=df_CL4_G3
RH=np.empty([len(df),37])*np.nan
U=np.empty([len(df),37])*np.nan
V=np.empty([len(df),37])*np.nan
T=np.empty([len(df),37])*np.nan
DP=np.empty([len(df),37])*np.nan
MX=np.empty([len(df),37])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G3=np.nanmean(H,axis=0)
rhum_CL4_G3=np.nanmean(RH,axis=0)
temp_CL4_G3=np.nanmean(T,axis=0)
dewp_CL4_G3=np.nanmean(DP,axis=0)
mixr_CL4_G3=np.nanmean(MX,axis=0)
v_CL4_G3=np.nanmean(V,axis=0)
u_CL4_G3=np.nanmean(U,axis=0)
wsp_CL4_G3=np.sqrt(u_CL4_G3**2 + v_CL4_G3**2)
wdir_CL4_G3=np.arctan2(-u_CL4_G3, -v_CL4_G3)*(180/np.pi)
wdir_CL4_G3[(u_CL4_G3 == 0) & (v_CL4_G3 == 0)]=0


dewp_CL4_G3[0]=np.nan

#*****************************************************************************\
#C4
#*****************************************************************************\
df=df_CL4_G4
RH=np.empty([len(df),37])*np.nan
U=np.empty([len(df),37])*np.nan
V=np.empty([len(df),37])*np.nan
T=np.empty([len(df),37])*np.nan
DP=np.empty([len(df),37])*np.nan
MX=np.empty([len(df),37])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G4=np.nanmean(H,axis=0)
rhum_CL4_G4=np.nanmean(RH,axis=0)
temp_CL4_G4=np.nanmean(T,axis=0)
dewp_CL4_G4=np.nanmean(DP,axis=0)
mixr_CL4_G4=np.nanmean(MX,axis=0)
v_CL4_G4=np.nanmean(V,axis=0)
u_CL4_G4=np.nanmean(U,axis=0)
wsp_CL4_G4=np.sqrt(u_CL4_G4**2 + v_CL4_G4**2)
wdir_CL4_G4=np.arctan2(-u_CL4_G4, -v_CL4_G4)*(180/np.pi)
wdir_CL4_G4[(u_CL4_G4 == 0) & (v_CL4_G4 == 0)]=0

dewp_CL4_G4[0]=np.nan


#*****************************************************************************\
#C5
#*****************************************************************************\
df=df_CL4_G5
RH=np.empty([len(df),37])*np.nan
U=np.empty([len(df),37])*np.nan
V=np.empty([len(df),37])*np.nan
T=np.empty([len(df),37])*np.nan
DP=np.empty([len(df),37])*np.nan
MX=np.empty([len(df),37])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G5=np.nanmean(H,axis=0)
rhum_CL4_G5=np.nanmean(RH,axis=0)
temp_CL4_G5=np.nanmean(T,axis=0)
dewp_CL4_G5=np.nanmean(DP,axis=0)
mixr_CL4_G5=np.nanmean(MX,axis=0)
v_CL4_G5=np.nanmean(V,axis=0)
u_CL4_G5=np.nanmean(U,axis=0)
wsp_CL4_G5=np.sqrt(u_CL4_G5**2 + v_CL4_G5**2)
wdir_CL4_G5=np.arctan2(-u_CL4_G5, -v_CL4_G5)*(180/np.pi)
wdir_CL4_G5[(u_CL4_G5 == 0) & (v_CL4_G5 == 0)]=0

dewp_CL4_G5[0]=np.nan

#*****************************************************************************\
#Plot
#*****************************************************************************\
#height_m=hlev_yotc
pressure_pa=pres_ei
#C1
height_m=hght_CL4_G1
temperature_c=temp_CL4_G1
dewpoint_c=dewp_CL4_G1
wsp=wsp_CL4_G1*1.943844
wdir=wdir_CL4_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 1')))

#C2
height_m=hght_CL4_G2
temperature_c=temp_CL4_G2
dewpoint_c=dewp_CL4_G2
wsp=wsp_CL4_G2*1.943844
wdir=wdir_CL4_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 2')))

#C3
height_m=hght_CL4_G3
temperature_c=temp_CL4_G3
dewpoint_c=dewp_CL4_G3
wsp=wsp_CL4_G3*1.943844
wdir=wdir_CL4_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 3')))
#C4
height_m=hght_CL4_G4
temperature_c=temp_CL4_G4
dewpoint_c=dewp_CL4_G4
wsp=wsp_CL4_G4*1.943844
wdir=wdir_CL4_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 4')))

#C5
height_m=hght_CL4_G5
temperature_c=temp_CL4_G5
dewpoint_c=dewp_CL4_G5
wsp=wsp_CL4_G5*1.943844
wdir=wdir_CL4_G5

mydataG5=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 5')))


#Individuals
S1=SkewT.Sounding(soundingdata=mydataG1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataG2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataG3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C3.png', format='png', dpi=1200)
S4=SkewT.Sounding(soundingdata=mydataG4)
S4.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C4.png', format='png', dpi=1200)
S5=SkewT.Sounding(soundingdata=mydataG5)
S5.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C5.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=0)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=1)
S.soundingdata=S4.soundingdata
S.add_profile(color='k',bloc=1)
S.soundingdata=S5.soundingdata
S.add_profile(color='m',bloc=2)
plt.savefig(path_data_save + '5K.png', format='png', dpi=1200)




#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              Closest Profiles 4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
pressure_pa=pres_ei

perc=0.25
#*****************************************************************************\
#C1
#*****************************************************************************\

df=df_CL4_G1.sort('dist_clus', ascending=True)
n_cases=int(round(len(df)*perc))

print df['dist_clus']
print n_cases, len(df)

RH=np.empty([n_cases,37])*np.nan
WSP=np.empty([n_cases,37])*np.nan
DIR=np.empty([n_cases,37])*np.nan
T=np.empty([n_cases,37])*np.nan
DP=np.empty([n_cases,37])*np.nan
MX=np.empty([n_cases,37])*np.nan
Dist1=np.empty([n_cases])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,n_cases):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist1[i]=np.array(df['dist_clus'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G1=np.nanmean(H,axis=0)
rhum_CL4_G1=np.nanmean(RH,axis=0)
temp_CL4_G1=np.nanmean(T,axis=0)
dewp_CL4_G1=np.nanmean(DP,axis=0)
mixr_CL4_G1=np.nanmean(MX,axis=0)
v_CL4_G1=np.nanmean(V,axis=0)
u_CL4_G1=np.nanmean(U,axis=0)
wsp_CL4_G1=np.sqrt(u_CL4_G1**2 + v_CL4_G1**2)
wdir_CL4_G1=np.arctan2(-u_CL4_G1, -v_CL4_G1)*(180/np.pi)
wdir_CL4_G1[(u_CL4_G1 == 0) & (v_CL4_G1 == 0)]=0
dewp_CL4_G1[0]=np.nan

#*****************************************************************************\
#Plot
height_m=hght_CL4_G1
#CF
temperature_c=temp_CL4_G1
dewpoint_c=dewp_CL4_G1
wsp=wsp_CL4_G1*1.943844
wdir=wdir_CL4_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 1 (25%)')))


S=SkewT.Sounding(soundingdata=mydataG1)
S.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C1_per.png', format='png', dpi=1200)

#*****************************************************************************\
#C2
#*****************************************************************************\

df=df_CL4_G2.sort('dist_clus', ascending=True)
n_cases=int(round(len(df)*perc))
RH=np.empty([n_cases,37])*np.nan
WSP=np.empty([n_cases,37])*np.nan
DIR=np.empty([n_cases,37])*np.nan
T=np.empty([n_cases,37])*np.nan
DP=np.empty([n_cases,37])*np.nan
MX=np.empty([n_cases,37])*np.nan
Dist2=np.empty([n_cases])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,n_cases):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist2[i]=np.array(df['dist_clus'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G2=np.nanmean(H,axis=0)
rhum_CL4_G2=np.nanmean(RH,axis=0)
temp_CL4_G2=np.nanmean(T,axis=0)
dewp_CL4_G2=np.nanmean(DP,axis=0)
mixr_CL4_G2=np.nanmean(MX,axis=0)
v_CL4_G2=np.nanmean(V,axis=0)
u_CL4_G2=np.nanmean(U,axis=0)
wsp_CL4_G2=np.sqrt(u_CL4_G2**2 + v_CL4_G2**2)
wdir_CL4_G2=np.arctan2(u_CL4_G2, v_CL4_G2)*(180/np.pi)+180
wdir_CL4_G2[(u_CL4_G2 == 0) & (v_CL4_G2 == 0)]=0
dewp_CL4_G2[0]=np.nan

#*****************************************************************************\
#Plot
height_m=hght_CL4_G2
#CF
temperature_c=temp_CL4_G2
dewpoint_c=dewp_CL4_G2
wsp=wsp_CL4_G2*1.943844
wdir=wdir_CL4_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 2 (25%)')))


S=SkewT.Sounding(soundingdata=mydataG2)
S.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C2_per.png', format='png', dpi=1200)

#*****************************************************************************\
#C3
#*****************************************************************************\

df=df_CL4_G3.sort('dist_clus', ascending=True)
n_cases=int(round(len(df)*perc))
RH=np.empty([n_cases,37])*np.nan
WSP=np.empty([n_cases,37])*np.nan
DIR=np.empty([n_cases,37])*np.nan
T=np.empty([n_cases,37])*np.nan
DP=np.empty([n_cases,37])*np.nan
MX=np.empty([n_cases,37])*np.nan
Dist3=np.empty([n_cases])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,n_cases):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist3[i]=np.array(df['dist_clus'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G3=np.nanmean(H,axis=0)
rhum_CL4_G3=np.nanmean(RH,axis=0)
temp_CL4_G3=np.nanmean(T,axis=0)
dewp_CL4_G3=np.nanmean(DP,axis=0)
mixr_CL4_G3=np.nanmean(MX,axis=0)
v_CL4_G3=np.nanmean(V,axis=0)
u_CL4_G3=np.nanmean(U,axis=0)
wsp_CL4_G3=np.sqrt(u_CL4_G3**2 + v_CL4_G3**2)
wdir_CL4_G3=np.arctan2(-u_CL4_G3, -v_CL4_G3)*(180/np.pi)
wdir_CL4_G3[(u_CL4_G3 == 0) & (v_CL4_G3 == 0)]=0
dewp_CL4_G3[0]=np.nan

#*****************************************************************************\
#Plot
height_m=hght_CL4_G3

#CF
temperature_c=temp_CL4_G3
dewpoint_c=dewp_CL4_G3
wsp=wsp_CL4_G3*1.943844
wdir=wdir_CL4_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 3 (25%)')))


S=SkewT.Sounding(soundingdata=mydataG3)
S.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C3_per.png', format='png', dpi=1200)

#*****************************************************************************\
#C4
#*****************************************************************************\

df=df_CL4_G4.sort('dist_clus', ascending=True)
n_cases=int(round(len(df)*perc))
RH=np.empty([n_cases,37])*np.nan
WSP=np.empty([n_cases,37])*np.nan
DIR=np.empty([n_cases,37])*np.nan
T=np.empty([n_cases,37])*np.nan
DP=np.empty([n_cases,37])*np.nan
MX=np.empty([n_cases,37])*np.nan
Dist4=np.empty([n_cases])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,n_cases):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist4[i]=np.array(df['dist_clus'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G4=np.nanmean(H,axis=0)
rhum_CL4_G4=np.nanmean(RH,axis=0)
temp_CL4_G4=np.nanmean(T,axis=0)
dewp_CL4_G4=np.nanmean(DP,axis=0)
mixr_CL4_G4=np.nanmean(MX,axis=0)
v_CL4_G4=np.nanmean(V,axis=0)
u_CL4_G4=np.nanmean(U,axis=0)
wsp_CL4_G4=np.sqrt(u_CL4_G4**2 + v_CL4_G4**2)
wdir_CL4_G4=np.arctan2(-u_CL4_G4, -v_CL4_G4)*(180/np.pi)
wdir_CL4_G4[(u_CL4_G4 == 0) & (v_CL4_G4 == 0)]=0
dewp_CL4_G4[0]=np.nan

#*****************************************************************************\
#Plot
height_m=hght_CL4_G4
#CF
temperature_c=temp_CL4_G4
dewpoint_c=dewp_CL4_G4
wsp=wsp_CL4_G4*1.943844
wdir=wdir_CL4_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 4 (25%)')))


S=SkewT.Sounding(soundingdata=mydataG4)
S.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C4_per.png', format='png', dpi=1200)


#*****************************************************************************\
#C1
#*****************************************************************************\

df=df_CL4_G5.sort('dist_clus', ascending=True)
n_cases=int(round(len(df)*perc))

print df['dist_clus']
print n_cases, len(df)

RH=np.empty([n_cases,37])*np.nan
WSP=np.empty([n_cases,37])*np.nan
DIR=np.empty([n_cases,37])*np.nan
T=np.empty([n_cases,37])*np.nan
DP=np.empty([n_cases,37])*np.nan
MX=np.empty([n_cases,37])*np.nan
Dist1=np.empty([n_cases])*np.nan
H=np.empty([len(df),37])*np.nan


for i in range(0,n_cases):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist1[i]=np.array(df['dist_clus'][i])
    H[i,:]=np.array(df['height'][i])

#Mean profiles
hght_CL4_G5=np.nanmean(H,axis=0)
rhum_CL4_G5=np.nanmean(RH,axis=0)
temp_CL4_G5=np.nanmean(T,axis=0)
dewp_CL4_G5=np.nanmean(DP,axis=0)
mixr_CL4_G5=np.nanmean(MX,axis=0)
v_CL4_G5=np.nanmean(V,axis=0)
u_CL4_G5=np.nanmean(U,axis=0)
wsp_CL4_G5=np.sqrt(u_CL4_G5**2 + v_CL4_G5**2)
wdir_CL4_G5=np.arctan2(-u_CL4_G5, -v_CL4_G5)*(180/np.pi)
wdir_CL4_G5[(u_CL4_G5 == 0) & (v_CL4_G5 == 0)]=0
dewp_CL4_G5[0]=np.nan

#*****************************************************************************\
#Plot
height_m=hght_CL4_G5
#CF
temperature_c=temp_CL4_G5
dewpoint_c=dewp_CL4_G5
wsp=wsp_CL4_G5*1.943844
wdir=wdir_CL4_G5

mydataG5=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,' ','Cluster 5 (25%)')))


S=SkewT.Sounding(soundingdata=mydataG5)
S.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C5_per.png', format='png', dpi=1200)
