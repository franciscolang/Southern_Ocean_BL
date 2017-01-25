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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random as rd
import scipy.stats as st
import numpy as np
import scipy as sp
import scipy.stats
#from scipy import stats
from pylab import plot,show, grid, xlabel, ylabel, xlim, ylim, yticks, legend
from matplotlib import gridspec

base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/Main_Inv/'


Yfin=2011
n_rd=30 #Number of cases to take 65
n_rdyi=70 #Number of cases to take Yi 70
#*****************************************************************************\
#Default Infouarie Island, Australia', 'lat': -54.62, 'lon': 158.85}

ptemp_thold_main=0.010           # K/m
ptemp_thold_sec=0.005            # K/m
shear_thold=0.015
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            ERA-i Data
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_erai=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/'
matb1= sio.loadmat(path_data_erai+'ERAImac_1995.mat')
temp_erai=matb1['temp'][:]
rh_erai=matb1['rh'][:]
q_erai=matb1['q'][:]
u_erai=matb1['u'][:]
v_erai=matb1['v'][:]
time_erai= matb1['time2'][:]

#Pressure Levels
pres_erai=matb1['levels'][:] #hPa
pres_ei=pres_erai[0,::-1]


for y in range(1996,Yfin):
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
#*****************************************************************************\
#Variables
#*****************************************************************************\
#Calculate Potential Temperature
theta_ei=(temp_ei)*(1000./pres_ei)**0.287;

#Calculate Virtual Potential Temperature
thetav_ei=(1+0.61*(mixr_ei/1000.))*theta_ei;
tempv_ei=(1+0.61*(mixr_ei/1000.))*temp_ei;

#Calculate Wind Shear and Gradiente Potential Temp
dthetav_ei=np.zeros(theta_ei.shape)
vshear_ei=np.empty(theta_ei.shape)*np.nan

for j in range(0,len(time)):
    for i in range(1,len(pres_ei)-1):
        dthetav_ei[j,i]=(thetav_ei[j,i+1]-thetav_ei[j,i])/float(hght_ei[i+1]-hght_ei[i])
        vshear_ei[j,i]=np.sqrt(float((u_ei[j,i]-u_ei[j,i-1])**2+(v_ei[j,i]-v_ei[j,i-1])**2))/float(hght_ei[i]-hght_ei[i-1])

#Relative Humidity
rh_erai[rh_erai>100]=100
rh_erai[rh_erai<0]=0

# plot(thetav_ei[:,1],'b',temp_ei[:,1],'r')
# show()
#******************************************************************************
#Boundary Layer Height Inversion 1 and 2
#Variables Initialization
sec_ind=np.empty(len(time))*np.nan
main_inv=np.empty(len(time),'float')*np.nan
sec_inv=np.empty(len(time))*np.nan
main_inversion=np.empty(len(time))*np.nan
sec_inversion=np.empty(len(time))*np.nan
main_inv_hght=np.empty(len(time))*np.nan
sec_inv_hght=np.empty(len(time))*np.nan

ei_hght_1inv=np.empty(len(time))*np.nan
ei_hght_2inv=np.empty(len(time))*np.nan
ei_strg_1inv=np.empty(len(time))*np.nan
ei_strg_2inv=np.empty(len(time))*np.nan

yotc_clas=np.empty(len(time))*np.nan



#******************************************************************************
#Main Inversion Position
for ind,line in enumerate(hght_ei):
    if line>=float(100.):
        twenty_y_index=ind
        break
for ind,line in enumerate(hght_ei):
    if line>=2500:
        twoky=ind
        break

main_inv=dthetav_ei[:,twenty_y_index:twoky].argmax(axis=1)
[i for i, j in enumerate(dthetav_ei[:,twenty_y_index:twoky]) if j == main_inv]
main_inv+=twenty_y_index #posicion main inv mas indice de sobre 100 m (3)

# main inversion must be > theta_v threshold
ptemp_comp1=dthetav_ei[:,main_inv[:]].diagonal() #extrae diagonal de pot temp

for i in range(0,len(time)):
    if ptemp_comp1[i]>ptemp_thold_main:
        main_inv_hght[i]=hght_ei[main_inv[i]]
        main_inversion[i]=True
        ei_strg_1inv[i]=dthetav_ei[i,main_inv[i]]
    else:
        main_inv_hght[i]=hght_ei[main_inv[i]]
        ei_strg_1inv[i]=np.nan

    #print main_inv_hght[i], ei_strg_1inv[i]


ei_inv_hght=main_inv_hght
ei_inv_strg=ei_strg_1inv
ei_clas=np.array([1]*len(ei_inv_hght))

# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
matb1= sio.loadmat(path_databom+'BOM_1995.mat')
bom_in=matb1['BOM_S'][:]
timesd= matb1['time'][:]
bom=bom_in

for y in range(1996,Yfin):
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
temp=bom[:,2,:].reshape(ni[0],ni[2])
mixr=bom[:,5,:].reshape(ni[0],ni[2])
wdir_initial=bom[:,6,:].reshape(ni[0],ni[2])
wspd=bom[:,7,:].reshape(ni[0],ni[2])
relh=bom[:,4,:].reshape(ni[0],ni[2])

u=wspd*(np.cos(np.radians(270-wdir_initial)))
v=wspd*(np.sin(np.radians(270-wdir_initial)))

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data ERA-i Levels
#*****************************************************************************\
#*****************************************************************************\

# hlev_yotc=hght_ei

# prutemp=np.empty((len(hlev_yotc),0))
# pruwspd=np.empty((len(hlev_yotc),0))
# pruwdir=np.empty((len(hlev_yotc),0))
# prumixr=np.empty((len(hlev_yotc),0))


# for j in range(0,ni[2]):
# #for j in range(0,100):
# #height initialization
#     x=hght[:,j]
#     x[-1]=np.nan
#     new_x=hlev_yotc
# #Interpolation YOTC levels
#     yt=temp[:,j]
#     rest=interp1d(x,yt)(new_x)
#     prutemp=np.append(prutemp,rest)

#     ym=mixr[:,j]
#     resm=interp1d(x,ym)(new_x)
#     prumixr=np.append(prumixr,resm)

# tempmac_ylev_2=prutemp.reshape(-1,len(hlev_yotc)).transpose()
# mixrmac_ylev_2=prumixr.reshape(-1,len(hlev_yotc)).transpose()

#*****************************************************************************\

# #Interpolation to YOTC Levels
temp_pres=np.zeros((len(pres_ei),ni[2]),'float')
mixr_pres=np.zeros((len(pres_ei),ni[2]),'float')
u_pres=np.zeros((len(pres_ei),ni[2]),'float')
v_pres=np.zeros((len(pres_ei),ni[2]),'float')
relh_pres=np.zeros((len(pres_ei),ni[2]),'float')

for j in range(0,ni[2]):

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yw=u[~np.isnan(u[:,j]),j]
    yd=v[~np.isnan(v[:,j]),j]
    yr=relh[~np.isnan(relh[:,j]),j]

    xp=pres[~np.isnan(temp[:,j]),j]

    temp_interp_pres=si.UnivariateSpline(xp[::-1],yt[::-1],k=5)
    mixr_interp_pres=si.UnivariateSpline(xp[::-1],ym[::-1],k=5)
    u_interp_pres=si.UnivariateSpline(xp[::-1],yw[::-1],k=5)
    v_interp_pres=si.UnivariateSpline(xp[::-1],yd[::-1],k=5)
    relh_interp_pres=si.UnivariateSpline(xp[::-1],yr[::-1],k=5)

    for ind in range(0,len(pres_ei)):
        temp_pres[ind,j]=temp_interp_pres(pres_ei[ind])
        mixr_pres[ind,j]=mixr_interp_pres(pres_ei[ind])
        u_pres[ind,j]=u_interp_pres(pres_ei[ind])
        v_pres[ind,j]=v_interp_pres(pres_ei[ind])
        relh_pres[ind,j]=relh_interp_pres(pres_ei[ind])


    relh_pres[relh_pres[:,j]>np.nanmax(yr),j]=np.nan
    relh_pres[relh_pres[:,j]<np.nanmin(yr),j]=np.nan

    temp_pres[temp_pres[:,j]>np.nanmax(yt),j]=np.nan
    temp_pres[temp_pres[:,j]<np.nanmin(yt),j]=np.nan

    u_pres[u_pres[:,j]>np.nanmax(yw),j]=np.nan
    u_pres[u_pres[:,j]<np.nanmin(yw),j]=np.nan
    v_pres[v_pres[:,j]>np.nanmax(yd),j]=np.nan
    v_pres[v_pres[:,j]<np.nanmin(yd),j]=np.nan

    mixr_pres[mixr_pres[:,j]>np.nanmax(ym),j]=np.nan
    mixr_pres[mixr_pres[:,j]<np.nanmin(ym),j]=np.nan

    del xp, yt, ym, yw, yd, yr

tempmac_ylev=temp_pres
umac_ylev=u_pres
vmac_ylev=v_pres
mixrmac_ylev=mixr_pres
relhmac_ylev=relh_pres


wspdmac_ylev=np.sqrt(umac_ylev**2 + vmac_ylev**2)
wdirmac_ylev=np.arctan2(-umac_ylev, -vmac_ylev)*(180/np.pi)
wdirmac_ylev[(umac_ylev == 0) & (vmac_ylev == 0)]=0


#*****************************************************************************\
#Initialization Variables
wdir_my=np.empty(tempmac_ylev.shape)

spec_hum_my=np.empty(tempmac_ylev.shape)
tempv_my=np.empty(tempmac_ylev.shape)
ptemp_v_my=np.empty(tempmac_ylev.shape)
ptemp_my=np.empty(tempmac_ylev.shape)
brn_mya=np.empty(tempmac_ylev.shape)

ucomp_my=np.empty(tempmac_ylev.shape)
vcomp_my=np.empty(tempmac_ylev.shape)

ucomp_initial_my=np.empty(tempmac_ylev.shape)
vcomp_initial_my=np.empty(tempmac_ylev.shape)

vert_shear_my=np.empty(tempmac_ylev.shape)*np.nan
ptemp_gmy=np.empty(tempmac_ylev.shape)*np.nan
ptemp_v_gmy=np.empty(tempmac_ylev.shape)*np.nan
main_my_inv=np.empty(ni[2])*np.nan
twenty_my_index=[]
twokmy=[]

ptemp_comp2=np.empty(ni[2])*np.nan
main_my_inv_hght=np.empty(ni[2])*np.nan
main_my_inversion=np.empty(ni[2])*np.nan

mac_y_clas=np.empty(ni[2])*np.nan

mac_y_strg_1inv=np.empty(ni[2])*np.nan
main_my_inv=np.empty(ni[2])*np.nan
main_my_inversion=np.empty(ni[2])*np.nan
main_my_inv_hght=np.empty(ni[2])*np.nan

#*****************************************************************************\


mixrmac_ylev_2=mixrmac_ylev
tempmac_ylev_2=tempmac_ylev

for j in range(0,ni[2]):
#Calculate new variables
    for i in range(0,len(hght_ei)):
        spec_hum_my[i,j]=(float(mixrmac_ylev_2[i,j])/1000.)/(1+(float(mixrmac_ylev_2[i,j])/1000.))

        tempv_my[i,j]=tempmac_ylev_2[i,j]*float(1+0.61*spec_hum_my[i,j])

        ptemp_my[i,j]=(tempmac_ylev_2[i,j]+273.16)*((1000./pres_ei[i])**0.286)

        ptemp_v_my[i,j]=(tempv_my[i,j]+273.16)*((1000./pres_ei[i])**0.286)


    for i in range(0,len(hght_ei)-1):
        vert_shear_my[i,j]=np.sqrt(float((umac_ylev[i,j]-umac_ylev[i-1,j])**2+(vmac_ylev[i,j]-vmac_ylev[i-1,j])**2))/float(hght_ei[i+1]-hght_ei[i])

        ptemp_v_gmy[i,j]=(ptemp_v_my[i+1,j]-ptemp_v_my[i,j])/float(hght_ei[i+1]-hght_ei[i])

    vert_shear_my[-1,j]=np.nan
    ptemp_v_gmy[-1,j]=np.nan
    ptemp_v_gmy[0,:]=np.nan #Hace cero el valor de superficie

    #print ptemp_v_gmy[i,j]
#*****************************************************************************\
hlev_yotc=hght_ei

# #Main Inversion Position
for ind,line in enumerate(hlev_yotc):
    if line>=float(100.):
        twenty_my_index=ind
        break
for ind,line in enumerate(hlev_yotc):
    if line>=2500:
        twokmy=ind
        break

for j in range(0,ni[2]):
    if np.all(np.isnan(ptemp_v_gmy[twenty_my_index:twokmy,j])):
        main_my_inv[j]=0
    else:
        main_my_inv[j]=np.nanargmax(ptemp_v_gmy[twenty_my_index:twokmy,j],axis=0)



main_my_inv+=twenty_my_index #posicion main inv mas indice de sobre 100 m (3)

# main inversion must be > theta_v threshold
for j in range(0,ni[2]):
    ptemp_comp2[j]=ptemp_v_gmy[main_my_inv[j],j]

    if ptemp_comp2[j]>ptemp_thold_main:
        main_my_inv_hght[j]=hlev_yotc[main_my_inv[j]]
        mac_y_strg_1inv[j]=ptemp_comp2[j]
        main_my_inversion[j]=True
    else:
        main_my_inversion[j]=False
        mac_y_strg_1inv[j]=np.nan


    #print  mac_y_strg_1inv[j],  main_my_inv_hght[j], ptemp_comp2[j]



mac_inv_hght=main_my_inv_hght
mac_inv_strg=mac_y_strg_1inv
mac_clas=np.array([1]*len(mac_inv_hght))

relhum_my=relhmac_ylev.T
temp_my=tempmac_ylev.T+273.16
u_my=umac_ylev.T
v_my=vmac_ylev.T
mixr_my=mixrmac_ylev.T
pot_temp_v_my=ptemp_v_my.T
pot_temp_my=ptemp_my.T
dthetav_my=ptemp_v_gmy.T
vertshear_my=vert_shear_my.T
q_my=spec_hum_my.T*1000
#tempv_my=tempv_my.T+273.16

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Reading Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'



#*****************************************************************************\
#Reading CSV
#*****************************************************************************\
df_front= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')
#*****************************************************************************\
df_yotc= pd.read_csv(path_data_csv + 'MCR/df_era_19952010.csv', sep='\t', parse_dates=['Date'])
df_my= pd.read_csv(path_data_csv + 'MCR/df_macera_19952010_5k.csv', sep='\t', parse_dates=['Date'])

df_yotc= df_yotc.set_index('Date')
df_my= df_my.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

df_yotcfro['Dist Front']=df_yotcfro['Dist CFront']*-1
df_myfro['Dist Front']=df_myfro['Dist CFront']*-1

df_yotcfro[(df_yotcfro['Dist Front']>10)]=np.nan
df_yotcfro[(df_yotcfro['Dist Front']<-10)]=np.nan
df_myfro[(df_myfro['Dist Front']>10)]=np.nan
df_myfro[(df_myfro['Dist Front']<-10)]=np.nan


#*****************************************************************************\
#Clasification by type
#*****************************************************************************\

df_1i= df_yotcfro[np.isfinite(df_yotcfro['1ra Inv'])]
df_yotc_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]


df_1s= df_yotcfro[np.isfinite(df_yotcfro['Strg 1inv'])]
df_yotc_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


#*****************************************************************************\
df_1i= df_myfro[np.isfinite(df_myfro['1ra Inv'])]
df_my_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_1s= df_myfro[np.isfinite(df_myfro['Strg 1inv'])]
df_my_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


#*****************************************************************************\
#Percentages
#*****************************************************************************\
#Total Number Sounding at YOTC of 1460
n_yotcsound=len(df_yotcfro[np.isfinite(df_yotcfro['Clas'])])
#Total Number Sounding at MAC of 3652
n_mysound=len(df_myfro[np.isfinite(df_myfro['Clas'])])

#Sounding YOTC
df_yotcsound=df_yotcfro[np.isfinite(df_yotcfro['Clas'])]
#Sounding MAC-YOTC
df_mysound=df_myfro[np.isfinite(df_myfro['Clas'])]


#Total Number Sounding at YOTC with front
n_yotcfronts=len(df_yotcsound[np.isfinite(df_yotcsound['Dist Front'])])
#Total Number Sounding at MAC-YOTC with front
n_myfronts=len(df_mysound[np.isfinite(df_mysound['Dist Front'])])

#*****************************************************************************\
#Plot
#*****************************************************************************\


x1=np.array(df_yotc_1inv['Dist Front'])
y=np.array(df_yotc_1inv['1ra Inv'])

bin_means_era, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)



x1=np.array(df_my_1inv['Dist Front'])
y=np.array(df_my_1inv['1ra Inv'])
bin_means_mac, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_mac, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)


bin_edges=np.arange(-9.5, 10.5, 1)


fig=plt.figure(figsize=(10, 6))
ax0=fig.add_subplot(111)
ax0.plot(bin_edges,bin_means_mac,'-o', label='MAC')
ax0.plot(bin_edges,bin_means_era,'-or', label='ERA-i')

ax0.errorbar(bin_edges, bin_means_mac, yerr=bin_std_mac, fmt='-ob',label='MAC')
ax0.errorbar(bin_edges, bin_means_era, yerr=bin_std_era, fmt='-or',label='ERA-i')

ax0.legend(loc=3,fontsize = 10, numpoints=1)
ax0.grid()
plt.savefig(path_data_save + 'heights.eps', format='eps', dpi=1200)

#*****************************************************************************\
# Rescale
#*****************************************************************************\
OldMax=0
OldMin=5000
NewMax=0
NewMin=20


OldValue_era=bin_means_era
OldValue_mac=bin_means_mac

OldValue_era_std=bin_std_era
OldValue_mac_std=bin_std_mac


OldRange = (OldMax - OldMin)
NewRange = (NewMax - NewMin)

NewValue_era = (((OldValue_era - OldMin) * NewRange) / OldRange) + NewMin
NewValue_mac = (((OldValue_mac - OldMin) * NewRange) / OldRange) + NewMin

NewValue_era_std = (((OldValue_era_std - OldMin) * NewRange) / OldRange) + NewMin
NewValue_mac_std = (((OldValue_mac_std - OldMin) * NewRange) / OldRange) + NewMin

fig=plt.figure(figsize=(8, 6))
ax0=fig.add_subplot(111)
ax0.plot(bin_edges,NewValue_era,'-o', label='MAC')
#ax0.plot(bin_edges,bin_means_era,'-or', label='ERA-i')
ax0.legend(loc=3,fontsize = 10)
ax0.grid()
#plt.show()

#*****************************************************************************\
#*****************************************************************************\
#                                   Cambiar fechas #*****************************************************************************\
#*****************************************************************************\
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
time_my_ori = np.array(timestamp)

for i in range(0,ni[2]):
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
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Date index del periodo 2006-2010
date_index_12h = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#date_index_12h = pd.date_range('1995-01-01 00:00', periods=1462, freq='12H')
# #*****************************************************************************\
#Dataframe ERA-interim
t_list=temp_ei.tolist()
u_list=u_ei.tolist()
v_list=v_ei.tolist()
rh_list=rh_ei.tolist()
q_list=q_ei.tolist()
mr_list=mixr_ei.tolist()
thetav_list=thetav_ei.tolist()
theta_list=theta_ei.tolist()
dthetav_list=dthetav_ei.tolist()
vertshear_list=vshear_ei.tolist()

dy={'Clas':ei_clas,
# 'MI Hght':ei_inv_hght,
# 'MI Strg':ei_inv_strg,
'temp':t_list,
'thetav':thetav_list,
'theta':theta_list,
'dthetav':dthetav_list,
'vertshear':vertshear_list,
'u':u_list,
'v':u_list,
'rh':rh_list,
'q':q_list,
'mixr':mr_list}
# 'Hght': ei_inv_hght,
# 'Strg': ei_inv_strg}

df_ei = pd.DataFrame(data=dy,index=date_erai)
df_ei.index.name = 'Date'
#*****************************************************************************\
#Dataframe ei All
df_erai=df_ei.reindex(date_index_12h)
df_erai.index.name = 'Date'


#*****************************************************************************\
dyc={'Clas ERA':ei_clas,
# 'MI Hght ERA':ei_inv_hght,
# 'MI Strg ERA':ei_inv_strg,
'temp ERA':t_list,
'thetav ERA':thetav_list,
'theta ERA':theta_list,
'dthetav ERA':dthetav_list,
'vertshear ERA':vertshear_list,
'u ERA':u_list,
'v ERA':u_list,
'rh ERA':rh_list,
'q ERA':q_list,
'mixr ERA':mr_list}
# 'Hght ERA': ei_inv_hght,
# 'Strg ERA': ei_inv_strg}

dfc_ei = pd.DataFrame(data=dyc,index=date_erai)
dfc_ei.index.name = 'Date'

dfc_era=dfc_ei.reindex(date_index_12h)
dfc_era.index.name = 'Date'

#*****************************************************************************\
#*****************************************************************************\#*****************************************************************************\
#*****************************************************************************\
#Dataframe MAC ERA-i levels
t_list=temp_my.tolist()
u_list=u_my.tolist()
v_list=v_my.tolist()
rh_list=relhum_my.tolist()
mr_list=mixr_my.tolist()
q_list=q_my.tolist()
theta_list=pot_temp_my.tolist()
thetav_list=pot_temp_v_my.tolist()
dthetav_list=dthetav_my.tolist()
vertshear_list=vertshear_my.tolist()

dmy={'Clas':mac_clas,
# 'MI Hght':mac_inv_hght,
# 'MI Strg':mac_inv_strg,
'temp':t_list,
'thetav':thetav_list,
'theta':theta_list,
'dthetav':dthetav_list,
'vertshear':vertshear_list,
'u':u_list,
'v':u_list,
'rh':rh_list,
'q':q_list,
'mixr':mr_list}
# 'Hght': mac_inv_hght,
# 'Strg': mac_inv_strg}

df_mac_y = pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
dfmy=df_mac_y.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_macei=dfmy.reindex(date_index_12h)
df_macei.index.name = 'Date'

#*****************************************************************************\

dmc={'Clas MAC':mac_clas,
# 'MI Hght MAC':mac_inv_hght,
# 'MI Strg MAC':mac_inv_strg,
'temp MAC':t_list,
'thetav MAC':thetav_list,
'theta MAC':theta_list,
'dthetav MAC':dthetav_list,
'vertshear MAC':vertshear_list,
'u MAC':u_list,
'v MAC':u_list,
'rh MAC':rh_list,
'q MAC':q_list,
'mixr MAC':mr_list}
# 'Hght': mac_inv_hght,
# 'Strg': mac_inv_strg}


dfc_m = pd.DataFrame(data=dmc,index=time_my)
# Eliminate Duplicate Soundings
dfc_m=dfc_m.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

dfc_mac=dfc_m.reindex(date_index_12h)
dfc_mac.index.name = 'Date'


#*****************************************************************************\

t_list=temp.T.tolist()
pres_list=pres.T.tolist()

#No interpol
dmc2={'pres':pres_list,
'temp':t_list}



dfc_m2 = pd.DataFrame(data=dmc2,index=time_my)
# Eliminate Duplicate Soundings
dfc_m2=dfc_m2.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

dfc_macnint=dfc_m2.reindex(date_index_12h)
dfc_macnint.index.name = 'Date'


#*****************************************************************************\
#*****************************************************************************\
#Combination ERA-i and MAC (Esta es para tomar solo casos donde hay mediciones de ambos el mismo dia y hora)

dfc_macera=pd.concat([dfc_mac,dfc_era],axis=1)

print np.count_nonzero(~np.isnan(dfc_macera['Clas ERA'])),  np.count_nonzero(~np.isnan(dfc_macera['Clas MAC']))
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Reading Fronts File
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

df_front= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')

#Merge datraframe mac with
df_eraifro=pd.concat([df_erai, df_front],axis=1)
df_meifro=pd.concat([df_macei, df_front],axis=1)

df_macerafro=pd.concat([dfc_macera, df_front],axis=1)
#Merge datraframe mac with no interpol

df_macni_fro=pd.concat([dfc_macnint, df_front],axis=1)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Plot Comparing
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# from matplotlib.projections import register_projection
# from skewx_projection_matplotlib_lt_1d4 import SkewXAxes
# register_projection(SkewXAxes)

# nc=700
# Tmacni=np.array(df_macni_fro['temp'][nc])
# Pmacni=np.array(df_macni_fro['pres'][nc])
# Tmac=np.array(df_macerafro['temp MAC'][nc])-273.16
# Pmac=pres_ei

# Tera=np.array(df_macerafro['temp ERA'][nc])-273.16
# Pera=pres_ei


# fig=plt.figure(figsize=(8, 6))
# ax0=fig.add_subplot(111, projection='skewx')
# plt.grid(True)
# ax0.semilogy(Tmacni, Pmacni, '-r',label='MAC Ori')
# ax0.semilogy(Tmac, Pmac, '.-g',label='MAC Int 1')
# ax0.semilogy(Tera, Pera, '.-b',label='ERA-i')

# l = ax0.axvline(0, color='b')

# ax0.yaxis.set_major_formatter(ScalarFormatter())
# ax0.set_yticks(np.linspace(100, 1000, 10))
# ax0.set_ylim(1050, 100)

# ax0.xaxis.set_major_locator(MultipleLocator(10))
# ax0.set_xlim(-60, 60)

# ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
# ax0.set_xlabel('Temperature (C)',fontsize = 10)
# plt.legend(loc=2,fontsize = 10)

# plt.show()



bins=np.arange(-10,11,1)
bx=np.arange(-10,10+1,5)
#*****************************************************************************\
    #Histogram
#*****************************************************************************\
df=df_meifro[np.isfinite(df_meifro['Clas'])]

df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
ncount_mac=pd.value_counts(df['catdist_fron']).sort_index()

y=np.array(ncount_mac)
x=np.array(ncount_mac.index.tolist())
x=x[::-1]

width=1


fig=plt.figure(figsize=(10, 6))
ax1=fig.add_subplot(111)

ax1.bar(x,y,width)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([0, 200])
ax1.set_yticklabels(np.arange(0,250,50), size=14)
ax1.axvline(0, color='k')
ax1.set_ylabel('Occurrences',fontsize = 16)
ax1.set_xlabel('Distance to front: cold to warm sector (deg)', size=16)
ax1.grid()
#ax1.margins(0.03)
plt.tight_layout()
plt.savefig(path_data_save + 'hist.eps', format='eps', dpi=1200)

plt.show()



#*****************************************************************************\
    #Histogram
#*****************************************************************************\

    # df=df_eraifro[np.isfinite(df_eraifro['Clas'])]

    # #df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
    # # ncount=pd.value_counts(df['catdist_fron'])
    # ncount_era=pd.value_counts(df['catdist_fron']).sort_index()

    # y=np.array(ncount_era)
    # x=np.array(ncount_era.index.tolist())
    # x=x[::-1]
    # ax1=fig.add_subplot(gs[1])

    # ax1.bar(x,y,width)

    # ax1.tick_params(axis='both', which='major', labelsize=12)
    # ax1.set_xticks(bx)
    # ax1.set_ylim([0, 200])
    # ax1.set_yticklabels(np.arange(0,250,50), size=12)
    # ax1.axvline(0, color='k')
    # ax1.set_ylabel('Occurrences',fontsize = 12)
    # ax1.set_xlabel('Distance to front: cold to warm sector (deg)', size=12)
    # ax1.grid()
    # ax1.margins(0.03)
    # plt.tight_layout()
    # plt.savefig(path_data_save + name_var[m]+'_ERA2.eps', format='eps', dpi=1200)
