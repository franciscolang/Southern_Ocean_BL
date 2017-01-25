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
#*****************************************************************************\
#*****************************************************************************\
#                            ERA-i Data
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_erai=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/'
matb1= sio.loadmat(path_data_erai+'ERAImac_2006.mat')
temp_erai=matb1['temp'][:]
rh_erai=matb1['rh'][:]
q_erai=matb1['q'][:]
u_erai=matb1['u'][:]
v_erai=matb1['v'][:]
time_erai= matb1['time2'][:]

#Pressure Levels
pres_erai=matb1['levels'][:] #hPa
pres_ei=pres_erai[0,::-1]


for y in range(2007,2011):
    matb= sio.loadmat(path_data_erai+'ERAImac_'+str(y)+'.mat')
    temp_r=matb['temp'][:]
    rh_r=matb['rh'][:]
    q_r=matb['q'][:]
    u_r=matb['u'][:]
    v_r=matb['v'][:]
    time_r= matb['time2'][:]

    if y==2010:
        rh_r=0.263*pres_ei*q_r*np.exp((17.67*(temp_r-273.16))/(temp_r-29.65))**(-1)*100
        rh_r[rh_r>100]=100
        rh_r[rh_r<0]=0

    temp_erai=np.concatenate((temp_erai,temp_r), axis=0)
    rh_erai=np.concatenate((rh_erai,rh_r), axis=0)
    q_erai=np.concatenate((q_erai,q_r), axis=0)
    v_erai=np.concatenate((v_erai,v_r), axis=0)
    u_erai=np.concatenate((u_erai,u_r), axis=0)
    time_erai=np.concatenate((time_erai,time_r), axis=1)

#*****************************************************************************\

#*****************************************************************************\
#Height Levels
file_levels = np.genfromtxt('./levels.csv', delimiter=',')
hght_ei=file_levels[:,2]*1000 #meters
#*****************************************************************************\
#Building dates from 1800-01-01-00:00
time=time_erai[0,:]
date_ini=datetime(1800, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Date Array
date_erai = pd.date_range(date_ini, periods=len(time), freq='6H')
#*****************************************************************************\
#Rotate from surface to free atmosphere
temp_ei=temp_erai[:,::-1]
u_ei=u_erai[:,::-1]
v_ei=v_erai[:,::-1]
q_ei=q_erai[:,::-1]*1000 #kg/kg=*1000 g/kg
mixr_ei= q_ei*1000 #g/kg
rh_ei=rh_erai[:,::-1]


e_ei=np.empty(rh_ei.shape)*np.nan
es_ei=np.empty(rh_ei.shape)*np.nan
dewp_ei=np.empty(rh_ei.shape)*np.nan

for j in range(0,len(time)):
    for i in range(0,len(pres_ei)):
        es_ei[j,i]=6.112 * np.exp((17.67 * (temp_ei[j,i]-273.16))/float((temp_ei[j,i]-273.16) + 243.5))
        e_ei[j,i] = es_ei[j,i] * (rh_ei[j,i]/float(100))
        dewp_ei[j,i] = np.log(e_ei[j,i]/float(6.112))*243.5/float(17.67-np.log(e_ei[j,i]/float(6.112)))+273.16

#*****************************************************************************\
Rd=287.04 #J/(kg K)
g=9.8 #m/s2
hght_ei_exp=np.empty(temp_ei.shape)
hght_ei_exp[:]=np.nan

for i in range(0,len(rh_ei)):
    for j in range(0,len(pres_ei)):
        hght_ei_exp[i,j]=(Rd*temp_ei[i,j])/float(g)*np.log(1000/float(pres_ei[j]))


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
pres_list=pres.T.tolist()
temp_list=temp.T.tolist()
mixr_list=mixr.T.tolist()
relh_list=relh.T.tolist()

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

#Dataframe ERA-interim
t_list=temp_ei.tolist()
u_list=u_ei.tolist()
v_list=v_ei.tolist()
rh_list=rh_ei.tolist()
q_list=q_ei.tolist()
mr_list=mixr_ei.tolist()
dp_list=dewp_ei.tolist()
hghtei_list=hght_ei_exp.tolist()

dyc={'temp ERA':t_list,
'dewp ERA':dp_list,
'u ERA':u_list,
'v ERA':v_list,
'RH ERA':rh_list,
'q ERA':q_list,
'mixr ERA':mr_list,
'height ERA':hghtei_list}

dfc_ei = pd.DataFrame(data=dyc,index=date_erai)
dfc_ei.index.name = 'Date'

dfc_era=dfc_ei.reindex(date_index_all)
dfc_era.index.name = 'Date'



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

#Unir datraframe mac-cyc con ERA
df_maceraclu=pd.concat([df_macclucyc, dfc_era],axis=1)

df_macclucyc = df_macclucyc[np.isfinite(df_macclucyc['dist_clus'])]
df_maceraclu = df_maceraclu[np.isfinite(df_maceraclu['dist_clus'])]



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

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/clusters/Cyclones/5K/'

#*****************************************************************************\
# 5 Clusters
#*****************************************************************************\

df_1all = df_maceraclu[df_maceraclu['cluster']==1]
df_2all = df_maceraclu[df_maceraclu['cluster']==2]
df_3all = df_maceraclu[df_maceraclu['cluster']==3]
df_4all = df_maceraclu[df_maceraclu['cluster']==4]
df_5all = df_maceraclu[df_maceraclu['cluster']==5]

df_1 = df_1all[np.isfinite(df_1all['dy'])]
df_2 = df_2all[np.isfinite(df_2all['dy'])]
df_3 = df_3all[np.isfinite(df_3all['dy'])]
df_4 = df_4all[np.isfinite(df_4all['dy'])]
df_5 = df_5all[np.isfinite(df_5all['dy'])]
#*****************************************************************************\


#*****************************************************************************\
#Cluster 1
#*****************************************************************************\

df=df_1.sort('dist_clus', ascending=True)

for i in range(0,3):
    RH=np.array(df['RH'][i])
    WSP=np.array(df['wsp'][i])
    DIR=np.array(df['wdir'][i])
    T=np.array(df['temp'][i])
    DP=np.array(df['dewp'][i])
    H=np.array(df['height'][i])
    date=df.index[i]
    #
    RHei=np.array(df['RH ERA'][i])
    Tei=np.array(df['temp ERA'][i])-273.16
    DPei=np.array(df['dewp ERA'][i])-273.16
    Hei=np.array(df['height ERA'][i])
    Uei=np.array(df['u ERA'][i])
    Vei=np.array(df['v ERA'][i])


    WSPei=np.sqrt(Uei**2 + Uei**2)
    DIRei=np.arctan2(Uei, Vei)*(180/np.pi)+180
    DIRei[(Uei == 0) & (Vei == 0)]=0
    DP[0]=np.nan

    #*****************************************************************************\
    height=H
    pressure=pres_ei
    temperature=T
    dewpoint=DP
    wsp=WSP*1.943844
    wdir=DIR

    mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC1 ('+str(date)+')')))

    S=SkewT.Sounding(soundingdata=mydataM1)

    #*****************************************************************************\
    height=Hei
    pressure=pres_ei
    temperature=Tei
    dewpoint=DPei
    wsp=WSPei*1.943844
    wdir=DIRei

    mydataE1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC1 ('+str(date)+')')))

    S2=SkewT.Sounding(soundingdata=mydataE1)


    S.make_skewt_axes()
    S.add_profile(color='r',bloc=0)
    S.soundingdata=S2.soundingdata
    S.add_profile(color='b',bloc=1)

    plt.savefig(path_data_save + '5kC1_C'+str(i+1)+'.eps', format='eps', dpi=1200)




#*****************************************************************************\
#Cluster 2
#*****************************************************************************\

df=df_2.sort('dist_clus', ascending=True)

for i in range(1,11):
    RH=np.array(df['RH'][i])
    WSP=np.array(df['wsp'][i])
    DIR=np.array(df['wdir'][i])
    T=np.array(df['temp'][i])
    DP=np.array(df['dewp'][i])
    H=np.array(df['height'][i])
    date=df.index[i]
    #
    RHei=np.array(df['RH ERA'][i])
    Tei=np.array(df['temp ERA'][i])-273.16
    DPei=np.array(df['dewp ERA'][i])-273.16
    Hei=np.array(df['height ERA'][i])
    Uei=np.array(df['u ERA'][i])
    Vei=np.array(df['v ERA'][i])


    WSPei=np.sqrt(Uei**2 + Uei**2)
    DIRei=np.arctan2(Uei, Vei)*(180/np.pi)+180
    DIRei[(Uei == 0) & (Vei == 0)]=0
    DP[0]=np.nan

    #*****************************************************************************\
    height=H
    pressure=pres_ei
    temperature=T
    dewpoint=DP
    wsp=WSP*1.943844
    wdir=DIR

    mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i) +' - 5KC2 ('+str(date)+')')))

    S=SkewT.Sounding(soundingdata=mydataM1)

    #*****************************************************************************\
    height=Hei
    pressure=pres_ei
    temperature=Tei
    dewpoint=DPei
    wsp=WSPei*1.943844
    wdir=DIRei

    mydataE1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i) +' - 5KC2 ('+str(date)+')')))

    S2=SkewT.Sounding(soundingdata=mydataE1)


    S.make_skewt_axes()
    S.add_profile(color='r',bloc=0)
    S.soundingdata=S2.soundingdata
    S.add_profile(color='b',bloc=1)

    plt.savefig(path_data_save + '5kC2_C'+str(i)+'.eps', format='eps', dpi=1200)


#*****************************************************************************\
#Cluster 3
#*****************************************************************************\

df=df_3.sort('dist_clus', ascending=True)

for i in range(0,4):
    RH=np.array(df['RH'][i])
    WSP=np.array(df['wsp'][i])
    DIR=np.array(df['wdir'][i])
    T=np.array(df['temp'][i])
    DP=np.array(df['dewp'][i])
    H=np.array(df['height'][i])
    date=df.index[i]
    #
    RHei=np.array(df['RH ERA'][i])
    Tei=np.array(df['temp ERA'][i])-273.16
    DPei=np.array(df['dewp ERA'][i])-273.16
    Hei=np.array(df['height ERA'][i])
    Uei=np.array(df['u ERA'][i])
    Vei=np.array(df['v ERA'][i])


    WSPei=np.sqrt(Uei**2 + Uei**2)
    DIRei=np.arctan2(Uei, Vei)*(180/np.pi)+180
    DIRei[(Uei == 0) & (Vei == 0)]=0
    DP[0]=np.nan

    #*****************************************************************************\
    height=H
    pressure=pres_ei
    temperature=T
    dewpoint=DP
    wsp=WSP*1.943844
    wdir=DIR

    mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC3 ('+str(date)+')')))

    S=SkewT.Sounding(soundingdata=mydataM1)

    #*****************************************************************************\
    height=Hei
    pressure=pres_ei
    temperature=Tei
    dewpoint=DPei
    wsp=WSPei*1.943844
    wdir=DIRei

    mydataE1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC3 ('+str(date)+')')))

    S2=SkewT.Sounding(soundingdata=mydataE1)


    S.make_skewt_axes()
    S.add_profile(color='r',bloc=0)
    S.soundingdata=S2.soundingdata
    S.add_profile(color='b',bloc=1)

    plt.savefig(path_data_save + '5kC3_C'+str(i+1)+'.eps', format='eps', dpi=1200)

#*****************************************************************************\
#Cluster 4
#*****************************************************************************\

df=df_4.sort('dist_clus', ascending=True)

for i in range(0,11):
    RH=np.array(df['RH'][i])
    WSP=np.array(df['wsp'][i])
    DIR=np.array(df['wdir'][i])
    T=np.array(df['temp'][i])
    DP=np.array(df['dewp'][i])
    H=np.array(df['height'][i])
    date=df.index[i]
    #
    RHei=np.array(df['RH ERA'][i])
    Tei=np.array(df['temp ERA'][i])-273.16
    DPei=np.array(df['dewp ERA'][i])-273.16
    Hei=np.array(df['height ERA'][i])
    Uei=np.array(df['u ERA'][i])
    Vei=np.array(df['v ERA'][i])


    WSPei=np.sqrt(Uei**2 + Uei**2)
    DIRei=np.arctan2(Uei, Vei)*(180/np.pi)+180
    DIRei[(Uei == 0) & (Vei == 0)]=0
    DP[0]=np.nan

    #*****************************************************************************\
    height=H
    pressure=pres_ei
    temperature=T
    dewpoint=DP
    wsp=WSP*1.943844
    wdir=DIR

    mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC4 ('+str(date)+')')))

    S=SkewT.Sounding(soundingdata=mydataM1)

    #*****************************************************************************\
    height=Hei
    pressure=pres_ei
    temperature=Tei
    dewpoint=DPei
    wsp=WSPei*1.943844
    wdir=DIRei

    mydataE1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC4 ('+str(date)+')')))

    S2=SkewT.Sounding(soundingdata=mydataE1)


    S.make_skewt_axes()
    S.add_profile(color='r',bloc=0)
    S.soundingdata=S2.soundingdata
    S.add_profile(color='b',bloc=1)

    plt.savefig(path_data_save + 'Cluster 4/5kC4_C'+str(i+1)+'.eps', format='eps', dpi=1200)

#*****************************************************************************\
#Cluster 5
#*****************************************************************************\

df=df_5.sort('dist_clus', ascending=True)

for i in range(0,3):
    RH=np.array(df['RH'][i])
    WSP=np.array(df['wsp'][i])
    DIR=np.array(df['wdir'][i])
    T=np.array(df['temp'][i])
    DP=np.array(df['dewp'][i])
    H=np.array(df['height'][i])
    date=df.index[i]
    #
    RHei=np.array(df['RH ERA'][i])
    Tei=np.array(df['temp ERA'][i])-273.16
    DPei=np.array(df['dewp ERA'][i])-273.16
    Hei=np.array(df['height ERA'][i])
    Uei=np.array(df['u ERA'][i])
    Vei=np.array(df['v ERA'][i])


    WSPei=np.sqrt(Uei**2 + Uei**2)
    DIRei=np.arctan2(Uei, Vei)*(180/np.pi)+180
    DIRei[(Uei == 0) & (Vei == 0)]=0
    DP[0]=np.nan

    #*****************************************************************************\
    height=H
    pressure=pres_ei
    temperature=T
    dewpoint=DP
    wsp=WSP*1.943844
    wdir=DIR

    mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC5 ('+str(date)+')')))

    S=SkewT.Sounding(soundingdata=mydataM1)

    #*****************************************************************************\
    height=Hei
    pressure=pres_ei
    temperature=Tei
    dewpoint=DPei
    wsp=WSPei*1.943844
    wdir=DIRei

    mydataE1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ','Case '+str(i+1) +' - 5KC5 ('+str(date)+')')))

    S2=SkewT.Sounding(soundingdata=mydataE1)


    S.make_skewt_axes()
    S.add_profile(color='r',bloc=0)
    S.soundingdata=S2.soundingdata
    S.add_profile(color='b',bloc=1)

    plt.savefig(path_data_save + '5kC5_C'+str(i+1)+'.eps', format='eps', dpi=1200)

