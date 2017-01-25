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
import scipy as sp
import scipy.stats
from scipy import array, arange, exp
from scipy.interpolate import interp1d
from scipy import interpolate
#from scipy import stats
from pylab import plot,show, grid, xlabel, ylabel, xlim, ylim, yticks, legend
from matplotlib import gridspec
from numpy import inf

base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/Main_Inv/'


Yfin=1997
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
file_levels = np.genfromtxt('../ERA-I/levels.csv', delimiter=',')
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

#np.count_nonzero(~np.isnan(data))

q=np.empty(mixr.shape)*np.nan
for i in range(0, len(timesd)):
    for j in range(0,3000):
        # q[j,i]=mixr[j,i]/(1+float(mixr[j,i])) #g/kg
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

tempv_my=np.empty(temp_pres.shape)
tempv_my[:]=np.nan
ptemp_v_my=np.empty(temp_pres.shape)
ptemp_v_my[:]=np.nan
ptemp_my=np.empty(temp_pres.shape)
ptemp_my=np.empty(temp_pres.shape)
ptemp_my[:]=np.nan

vert_shear_my=np.empty(temp_pres.shape)
vert_shear_my[:]=np.nan
ptemp_gmy=np.empty(temp_pres.shape)
ptemp_gmy[:]=np.nan
ptemp_v_gmy=np.empty(temp_pres.shape)
ptemp_v_gmy[:]=np.nan

twenty_my_index=[]
twokmy=[]

ptemp_comp2=np.empty(ni[1])
main_my_inv_hght=np.empty(ni[1])
main_my_inversion=np.empty(ni[1])

mac_y_clas=np.empty(ni[1])

mac_y_strg_1inv=np.empty(ni[1])
main_my_inv=np.empty(ni[1])
main_my_inversion=np.empty(ni[1])
main_my_inv_hght=np.empty(ni[1])
main_my_inv_hght=np.empty(ni[1])
main_my_inv_hght[:]=np.nan

spec_hum_my=q_pres
# spec_hum_my=np.empty(temp_pres.shape)
# spec_hum_my[:]=np.nan
#*****************************************************************************\

for j in range(0,ni[1]):
#Calculate new variables
    for i in range(0,len(hght_ei)):
        # spec_hum_my[i,j]=(float(mixr_pres[i,j])/1000.)/(1+(float(mixr_pres[i,j])/1000.))
        tempv_my[i,j]=temp_pres[i,j]*float(1+0.61*spec_hum_my[i,j])

        ptemp_my[i,j]=(temp_pres[i,j]+273.16)*((1000./pres_ei[i])**0.286)

        ptemp_v_my[i,j]=(tempv_my[i,j]+273.16)*((1000./pres_ei[i])**0.286)


    for i in range(0,len(hght_ei)-1):
        vert_shear_my[i,j]=np.sqrt(float((u_pres[i,j]-u_pres[i-1,j])**2+(v_pres[i,j]-v_pres[i-1,j])**2))/float(hght_ei[i+1]-hght_ei[i])

        ptemp_v_gmy[i,j]=(ptemp_v_my[i+1,j]-ptemp_v_my[i,j])/float(hght_ei[i+1]-hght_ei[i])

    vert_shear_my[-1,j]=np.nan
    ptemp_v_gmy[-1,j]=np.nan
    ptemp_v_gmy[0,:]=np.nan #Hace cero el valor de superficie

    vert_shear_my[vert_shear_my== inf] = np.nan

    ptemp_v_gmy[ptemp_v_gmy== inf] = np.nan
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

for j in range(0,ni[1]):
    if np.all(np.isnan(ptemp_v_gmy[twenty_my_index:twokmy,j])):
        main_my_inv[j]=0
    else:
        main_my_inv[j]=np.nanargmax(ptemp_v_gmy[twenty_my_index:twokmy,j],axis=0)

main_my_inv+=twenty_my_index #posicion main inv mas indice de sobre 100 m (3)

# main inversion must be > theta_v threshold
for j in range(0,ni[1]):
    ptemp_comp2[j]=ptemp_v_gmy[main_my_inv[j],j]

    if ptemp_comp2[j]>ptemp_thold_main:
        main_my_inv_hght[j]=hlev_yotc[main_my_inv[j]]
        print
        mac_y_strg_1inv[j]=ptemp_comp2[j]
        main_my_inversion[j]=True
    else:
        main_my_inversion[j]=False
        mac_y_strg_1inv[j]=np.nan


mac_inv_hght=main_my_inv_hght
mac_inv_strg=mac_y_strg_1inv
mac_clas=np.array([1]*len(mac_inv_hght))


ptemp_my[ptemp_my== inf] = np.nan
ptemp_my[ptemp_my== -inf] = np.nan
temp_pres[temp_pres== inf] = np.nan
temp_pres[temp_pres== -inf] = np.nan

relh_pres[relh_pres== inf] = np.nan
relh_pres[relh_pres== -inf] = np.nan
spec_hum_my[spec_hum_my== inf] = np.nan
spec_hum_my[spec_hum_my== -inf] = np.nan


#theta=pot_temp_my
relhum_my=relh_pres.T
temp_my=temp_pres.T+273.16
u_my=u_pres.T
v_my=v_pres.T
mixr_my=mixr_pres.T
pot_temp_v_my=ptemp_v_my.T
pot_temp_my=ptemp_my.T
dthetav_my=ptemp_v_gmy.T
vertshear_my=vert_shear_my.T
q_my=spec_hum_my.T*1000
dewp_my=dewp_pres.T+273.16
hght_my=hght_pres.T



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Plots
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# Specific Humidyt
#*****************************************************************************\

fig=plt.figure(figsize=(8, 6))
ax0=fig.add_subplot(111)
plt.grid(True)

case=0

VarMac=q[:,case]*1000
PMac=pres[:,case]

VarMacInt=q_my[case,:]
PMACInt=pres_ei

VarEra=q_ei[case,:]
PEra=pres_ei

ax0.semilogy(VarMac, PMac, '-r',label='MAC Ori')
ax0.semilogy(VarMacInt, PMACInt, '.-g',label='MAC Int')
#ax0.semilogy(VarEra, PEra, '.-b',label='ERA-i')


l = ax0.axvline(0, color='b')

ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.set_yticks(np.linspace(100, 1000, 10))
ax0.set_ylim(1050, 100)

ax0.xaxis.set_major_locator(MultipleLocator(2))
ax0.set_xlim(0, 10)

ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
ax0.set_xlabel('q (g kg^{-1})',fontsize = 10)
plt.legend(loc=2,fontsize = 10)




#*****************************************************************************\

fig=plt.figure(figsize=(8, 6))
ax0=fig.add_subplot(111)
plt.grid(True)

case=0

VarMac=q[:,case]*1000
PMac=hght[:,case]

VarMacInt=q_my[case,:]
PMACInt=hght_ei

VarEra=q_ei[case,:]
PEra=hght_ei

ax0.plot(VarMac, PMac, '-r',label='MAC Ori')
ax0.plot(VarMacInt, PMACInt, '.-g',label='MAC Int')
#ax0.semilogy(VarEra, PEra, '.-b',label='ERA-i')


l = ax0.axvline(0, color='b')

# ax0.yaxis.set_major_formatter(ScalarFormatter())
# ax0.set_yticks(np.linspace(100, 1000, 10))
# ax0.set_ylim(1050, 100)

ax0.xaxis.set_major_locator(MultipleLocator(2))
ax0.set_xlim(0, 10)

ax0.set_ylabel('Height (m)',fontsize = 10)
ax0.set_xlabel('q (g kg^{-1})',fontsize = 10)
plt.legend(loc=2,fontsize = 10)


#*****************************************************************************\
#height
#*****************************************************************************\

fig=plt.figure(figsize=(8, 6))
ax0=fig.add_subplot(111)
plt.grid(True)


ax0.semilogy(hght[:,case], PMac, '-r',label='MAC Ori')
ax0.semilogy(hght_pres[:,case], PMACInt, '.-g',label='MAC Int')

l = ax0.axvline(0, color='b')

ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.set_yticks(np.linspace(100, 1000, 10))
ax0.set_ylim(1050, 100)

# ax0.xaxis.set_major_locator(MultipleLocator(2))
# ax0.set_xlim(0, 10)

ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
#ax0.set_xlabel('q (g kg^{-1})',fontsize = 10)
plt.legend(loc=1,fontsize = 10)


plt.show()
