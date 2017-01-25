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
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/Interpolation/'


Yfin=2011
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
#Building dates from 1900-01-01-00:00
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

#Height Calculation
#*****************************************************************************\
Rd=287.04 #J/(kg K)
g=9.8 #m/s2
hght_ei_exp=np.empty(temp_ei.shape)
hght_ei_exp[:]=np.nan

for i in range(0,len(rh_ei)):
    for j in range(0,len(pres_ei)):
        hght_ei_exp[i,j]=(Rd*temp_ei[i,j])/float(g)*np.log(1000/float(pres_ei[j]))

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

q2=np.empty(mixr.shape)*np.nan
for i in range(0, len(timesd)):
    for j in range(0,3000):
        q2[j,i]=(float(mixr[j,i])/1000.)/(1+(float(mixr[j,i])/1000.)) #kg/kg


ni=pres.shape
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data ERA-i Levels (Interpolation)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\


#*****************************************************************************\
#*****************************************************************************\
#                                Linear Method
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
q2_pres=np.zeros((len(pres_ei),ni[1]),'float')
dewp_pres=np.zeros((len(pres_ei),ni[1]),'float')

#Linear Interpolation
for j in range(0,ni[1]):
    x=np.log(pres[~np.isnan(temp[:,j]),j])

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yu=u[~np.isnan(u[:,j]),j]
    yv=v[~np.isnan(v[:,j]),j]
    yr=relh[~np.isnan(relh[:,j]),j]
    yq=q2[~np.isnan(q2[:,j]),j]
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
    q2_pres[:,j]=extrap1d(f_iq)(np.log(pres_ei))
    temp_pres[:,j]=extrap1d(f_it)(np.log(pres_ei))
    mixr_pres[:,j]=extrap1d(f_im)(np.log(pres_ei))
    u_pres[:,j]=extrap1d(f_iu)(np.log(pres_ei))
    v_pres[:,j]=extrap1d(f_iv)(np.log(pres_ei))
    relh_pres[:,j]=extrap1d(f_ir)(np.log(pres_ei))
    dewp_pres[:,j]=extrap1d(f_id)(np.log(pres_ei))


del yt, ym, yu, yv, yr, yd, yq
#*****************************************************************************\
# Wind Speed and Direction
wspd_pres=np.sqrt(u_pres**2 + v_pres**2)
wdir_pres=np.arctan2(-u_pres, -v_pres)*(180/np.pi)
wdir_pres[(u_pres == 0) & (v_pres == 0)]=0


#*****************************************************************************\
#*****************************************************************************\
#                                Spline Method
#*****************************************************************************\
#*****************************************************************************\

temp_spl=np.zeros((len(pres_ei),ni[1]),'float')
mixr_spl=np.zeros((len(pres_ei),ni[1]),'float')
u_spl=np.zeros((len(pres_ei),ni[1]),'float')
v_spl=np.zeros((len(pres_ei),ni[1]),'float')
relh_spl=np.zeros((len(pres_ei),ni[1]),'float')
dewp_spl=np.zeros((len(pres_ei),ni[1]),'float')


for j in range(0,ni[1]):

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yu=u[~np.isnan(u[:,j]),j]
    yv=v[~np.isnan(v[:,j]),j]
    yr=relh[~np.isnan(relh[:,j]),j]
    yd=dewp[~np.isnan(dewp[:,j]),j]

    xp=pres[~np.isnan(temp[:,j]),j]

    temp_interp_spl=si.UnivariateSpline(xp[::-1],yt[::-1],k=5)
    mixr_interp_spl=si.UnivariateSpline(xp[::-1],ym[::-1],k=5)
    u_interp_spl=si.UnivariateSpline(xp[::-1],yu[::-1],k=5)
    v_interp_spl=si.UnivariateSpline(xp[::-1],yv[::-1],k=5)
    relh_interp_spl=si.UnivariateSpline(xp[::-1],yr[::-1],k=5)
    dewp_interp_spl=si.UnivariateSpline(xp[::-1],yd[::-1],k=5)

    for ind in range(0,len(pres_ei)):
        temp_spl[ind,j]=temp_interp_spl(pres_ei[ind])
        mixr_spl[ind,j]=mixr_interp_spl(pres_ei[ind])
        u_spl[ind,j]=u_interp_spl(pres_ei[ind])
        v_spl[ind,j]=v_interp_spl(pres_ei[ind])
        relh_spl[ind,j]=relh_interp_spl(pres_ei[ind])
        dewp_spl[ind,j]=dewp_interp_spl(pres_ei[ind])


    relh_spl[relh_spl[:,j]>np.nanmax(yr),j]=np.nan
    relh_spl[relh_spl[:,j]<np.nanmin(yr),j]=np.nan

    temp_spl[temp_spl[:,j]>np.nanmax(yt),j]=np.nan
    temp_spl[temp_spl[:,j]<np.nanmin(yt),j]=np.nan

    u_spl[u_spl[:,j]>np.nanmax(yu),j]=np.nan
    u_spl[u_spl[:,j]<np.nanmin(yu),j]=np.nan

    v_spl[v_spl[:,j]>np.nanmax(yv),j]=np.nan
    v_spl[v_spl[:,j]<np.nanmin(yv),j]=np.nan

    mixr_spl[mixr_spl[:,j]>np.nanmax(ym),j]=np.nan
    mixr_spl[mixr_spl[:,j]<np.nanmin(ym),j]=np.nan

    dewp_spl[dewp_spl[:,j]>np.nanmax(yd),j]=np.nan
    dewp_spl[dewp_spl[:,j]<np.nanmin(yd),j]=np.nan

    del xp, yt, yu, yv, yd, yr, ym

# tempmac_ylev=temp_spl
# umac_ylev=u_spl
# vmac_ylev=v_spl
# mixrmac_ylev=mixr_spl
# relhmac_ylev=relh_spl

wspd_spl=np.sqrt(u_spl**2 + v_spl**2)
wdir_spl=np.arctan2(-u_spl, -v_spl)*(180/np.pi)
wdir_spl[(u_spl == 0) & (v_spl == 0)]=0

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

#spec_hum_my=q_pres
spec_hum_my=np.empty(temp_pres.shape)
spec_hum_my[:]=np.nan

spec_hum_my_spl=np.empty(temp_pres.shape)
spec_hum_my_spl[:]=np.nan


#*****************************************************************************\

for j in range(0,ni[1]):
#Calculate new variables
    for i in range(0,len(hght_ei)):
        spec_hum_my[i,j]=(float(mixr_pres[i,j])/1000.)/(1+(float(mixr_pres[i,j])/1000.))

        spec_hum_my_spl[i,j]=(float(mixr_spl[i,j])/1000.)/(1+(float(mixr_spl[i,j])/1000.))


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

#*****************************************************************************\
#Height Calculation
#*****************************************************************************\

hght_my_exp=np.empty(temp_pres.shape)
hght_my_exp[:]=np.nan

for i in range(0,ni[1]):
    for j in range(0,len(pres_ei)):
        hght_my_exp[j,i]=(Rd*(temp_pres[j,i]+273.16))/float(g)*np.log(1000/float(pres_ei[j]))



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

#Linear
relhum_my=relh_pres.T
temp_my=temp_pres.T+273.16
u_my=u_pres.T
v_my=v_pres.T
mixr_my=mixr_pres.T
pot_temp_v_my=ptemp_v_my.T
pot_temp_my=ptemp_my.T
dthetav_my=ptemp_v_gmy.T
vertshear_my=vert_shear_my.T
q_my=spec_hum_my.T*1000 #g/kg
dewp_my=dewp_pres.T+273.16
hght_my=hght_my_exp.T

#Spline
temp_spl=temp_spl.T
u_spl=u_spl.T
v_spl=v_spl.T
mixr_spl=mixr_spl.T
relh_spl=relh_spl.T
dewp_spl=dewp_spl.T
q_spl=spec_hum_my_spl.T*1000 #g/kg




#*****************************************************************************\
for i in range(0, ni[1]):
    for j in range(0,37):
        if q_my[i,j]<0:
            q_my[i,:]=np.nan
        if q_my[i,j]>10:
            q_my[i,:]=np.nan

#np.nanargmax(np.nanmax(q_my,axis=1)) sounding con max


#*****************************************************************************\
#*****************************************************************************\
#           Plotting cases compared by interpolation method
#*****************************************************************************\
#*****************************************************************************\


# caso=10

# Td_ori=q2[:,caso]*1000
# Td_int=q_my[caso,:]
# Td_spl=q_spl[caso,:]

# p_ori=pres[:,caso]
# p_ei=pres_ei

# fig = plt.figure(figsize=(6.5875, 6.2125))
# ax = fig.add_subplot(111,)

# ax.semilogy(Td_ori, p_ori)
# ax.semilogy(Td_int, p_ei,'r')
# ax.semilogy(Td_spl, p_ei, 'g')

# l = ax.axvline(0, color='b')
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.set_yticks(np.linspace(100, 1000, 10))
# ax.set_ylim(1050, 100)
# plt.grid(True)



#*****************************************************************************\
#*****************************************************************************\
#                                   Cambiar fechas #*****************************************************************************\
#*****************************************************************************\
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
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
date_index_12h = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#date_index_12h = pd.date_range('1995-01-01 00:00', periods=1462, freq='12H')
# #*****************************************************************************\
#Dataframe ERA-interim
rh_list=rh_ei.tolist()
q_list=q_ei.tolist()
mr_list=mixr_ei.tolist()
hght_ei_list=hght_ei_exp.tolist()


dy={'rh ERA':rh_list,
'q ERA':q_list,
'mixr ERA':mr_list,
'hght ERA': hght_ei_list}


df_ei = pd.DataFrame(data=dy,index=date_erai)
df_ei.index.name = 'Date'

df_erai=df_ei.reindex(date_index_12h)
df_erai.index.name = 'Date'


#*****************************************************************************\
#*****************************************************************************\#*****************************************************************************\
#*****************************************************************************\
#Dataframe MAC ERA-i levels
#Linear
rh_lin_list=relhum_my.tolist()
mr_lin_list=mixr_my.tolist()
q_lin_list=q_my.tolist()
hght_my_list=hght_my.tolist()
#Spline
rh_spl_list=relh_spl.tolist()
mr_spl_list=mixr_spl.tolist()
q_spl_list=q_spl.tolist()
#Original
q_ori_list=q2.T.tolist()
pres_ori_list=pres.T.tolist()
hght_ori_list=hght.T.tolist()

dmy={'pres MAC':pres_ori_list,
'hght MAC':hght_ori_list,
'q MAC':q_ori_list,
'rh LMAC':rh_lin_list,
'q LMAC':q_lin_list,
'mixr LMAC':mr_lin_list,
'rh SMAC':rh_spl_list,
'q SMAC':q_spl_list,
'mixr SMAC':mr_spl_list,
'hght LMAC': hght_my_list}


df_mac_y = pd.DataFrame(data=dmy,index=time_my)
df_mac_y.index.name = 'Date'

dfmy = df_mac_y[~df_mac_y.index.duplicated(keep='first')]

df_mac=dfmy.reindex(date_index_12h)
df_mac.index.name = 'Date'


#*****************************************************************************\
#*****************************************************************************\
#Combination ERA-i and MAC (Esta es para tomar solo casos donde hay mediciones de ambos el mismo dia y hora)

dfc_macera=pd.concat([df_mac,df_erai],axis=1)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                   Reading Fronts File and Merge
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'


df_front= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')

df_front=df_front.reindex(date_index_12h)
df_front.index.name = 'Date'


#Merge datraframe mac with
df_eraifro=pd.concat([df_erai, df_front],axis=1)
df_meifro=pd.concat([df_mac, df_front],axis=1)

df_macerafro=pd.concat([dfc_macera, df_front],axis=1)

#*****************************************************************************\
#*****************************************************************************\

df=df_macerafro
bins=np.arange(-10,11,1)

df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
ncount=pd.value_counts(df['catdist_fron'])

df_all=df

print ncount

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                Plotting cases compared by interpolation method
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

#df_all.ix['1996-01-01 00'] Identificar fecha especifica
#df_all.loc['1996-01-01 00']

# for j in range(3,10):

#     df_plot1=df_all[(df_all['catdist_fron']==j)]

#     y=[0,10,20,35,60]
#     #y=[0,1,4]

#     for i in range(0,len(y)):
#         Q_ori=np.array(df_plot1['q MAC'][y[i]])*1000
#         Q_spl=np.array(df_plot1['q SMAC'][y[i]])
#         Q_lin=np.array(df_plot1['q LMAC'][y[i]])
#         Q_era=np.array(df_plot1['q ERA'][y[i]])

#         p_ori=np.array(df_plot1['pres MAC'][y[i]])

#         H_ori=np.array(df_plot1['hght MAC'][y[i]])
#         H_era=np.array(df_plot1['hght ERA'][y[i]])
#         H_int=np.array(df_plot1['hght LMAC'][y[i]])


# #*****************************************************************************\
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
#         #ax = fig.add_subplot(111,)
# #*****************************************************************************\
#         #Pressure Levels
#         ax1.semilogy(Q_ori, p_ori,label='MAC original',linewidth=2)
#         ax1.semilogy(Q_lin, pres_ei,'r',label='MAC lin_int',linewidth=2)
#         ax1.semilogy(Q_spl, pres_ei, 'g',label='MAC spl_int',linewidth=2)
#         ax1.semilogy(Q_era, pres_ei, 'y',label='ERAi',linewidth=2)

#         l = ax1.axvline(0, color='b')
#         ax1.yaxis.set_major_formatter(ScalarFormatter())
#         ax1.set_yticks(np.arange(100, 1050, 50))
#         ax1.set_ylim(1000, 600)
#         ax1.set_xlim(0, 6)
#         ax1.legend(loc=1,fontsize = 10)
#         ax1.grid(True)

#         ax1.set_title('Example - ' + str(i+1), size=12)
#         ax1.set_ylabel('Pressure (hPa)')
#         ax1.set_xlabel('q (g kg$^{-1}$)')

# #*****************************************************************************\
#         #Altitute
#         ax2.plot(Q_ori, H_ori,label='MAC original',linewidth=2)
#         ax2.plot(Q_lin, H_int,'r',label='MAC lin_int',linewidth=2)
#         ax2.plot(Q_spl, H_int, 'g',label='MAC spl_int',linewidth=2)
#         ax2.plot(Q_era, H_era, 'y',label='ERAi',linewidth=2)

#         #ax2.set_yticks(np.arange(100, 1050, 50))
#         ax2.set_ylim(0, 5000)
#         ax2.set_xlim(0, 6)
#         ax2.grid(True)

#         ax2.set_ylabel('Altitude (m)')
#         ax2.set_xlabel('q (g kg$^{-1}$)')


#         plt.savefig(path_data_save + 'examples/indiv_post_example' + str(i+1) + '_col' + str(j) +'.eps', format='eps', dpi=300)


# for j in range(3,10):

#     ji=j*(-1)

#     df_plot1=df_all[(df_all['catdist_fron']==ji)]

#     y=[1,10,20,35,60]
#     #y=[0,2,5]

#     for i in range(0,len(y)):
#         Q_ori=np.array(df_plot1['q MAC'][y[i]])*1000
#         Q_spl=np.array(df_plot1['q SMAC'][y[i]])
#         Q_lin=np.array(df_plot1['q LMAC'][y[i]])
#         Q_era=np.array(df_plot1['q ERA'][y[i]])

#         p_ori=np.array(df_plot1['pres MAC'][y[i]])

#         H_ori=np.array(df_plot1['hght MAC'][y[i]])
#         H_era=np.array(df_plot1['hght ERA'][y[i]])
#         H_int=np.array(df_plot1['hght LMAC'][y[i]])


# #*****************************************************************************\
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
#         #ax = fig.add_subplot(111,)
# #*****************************************************************************\
#         #Pressure Levels
#         ax1.semilogy(Q_ori, p_ori,label='MAC original',linewidth=2)
#         ax1.semilogy(Q_lin, pres_ei,'r',label='MAC lin_int',linewidth=2)
#         ax1.semilogy(Q_spl, pres_ei, 'g',label='MAC spl_int',linewidth=2)
#         ax1.semilogy(Q_era, pres_ei, 'y',label='ERAi',linewidth=2)

#         l = ax1.axvline(0, color='b')
#         ax1.yaxis.set_major_formatter(ScalarFormatter())
#         ax1.set_yticks(np.arange(100, 1050, 50))
#         ax1.set_ylim(1000, 600)
#         ax1.set_xlim(0, 6)
#         ax1.legend(loc=1,fontsize = 10)
#         ax1.grid(True)

#         ax1.set_title('Example - ' + str(i+1), size=12)
#         ax1.set_ylabel('Pressure (hPa)')
#         ax1.set_xlabel('q (g kg$^{-1}$)')

# #*****************************************************************************\
#         #Altitute
#         ax2.plot(Q_ori, H_ori,label='MAC original',linewidth=2)
#         ax2.plot(Q_lin, H_int,'r',label='MAC lin_int',linewidth=2)
#         ax2.plot(Q_spl, H_int, 'g',label='MAC spl_int',linewidth=2)
#         ax2.plot(Q_era, H_era, 'y',label='ERAi',linewidth=2)

#         #ax2.set_yticks(np.arange(100, 1050, 50))
#         ax2.set_ylim(0, 5000)
#         ax2.set_xlim(0, 6)
#         ax2.grid(True)

#         ax2.set_ylabel('Altitude (m)')
#         ax2.set_xlabel('q (g kg$^{-1}$)')


#         plt.savefig(path_data_save + 'examples/indiv_pre_example' + str(i+1) + '_col' + str(j) +'.eps', format='eps', dpi=300)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                                   Profiles and Fronts
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# Means Complete Period
#*****************************************************************************\

MG_macS=np.empty([len(ncount),len(pres_ei)])*np.nan
RH_macS=np.empty([max(ncount),len(pres_ei)])*np.nan
MG_macL=np.empty([len(ncount),len(pres_ei)])*np.nan
RH_macL=np.empty([max(ncount),len(pres_ei)])*np.nan

MG_era=np.empty([len(ncount),len(pres_ei)])*np.nan
RH_era=np.empty([max(ncount),len(pres_ei)])*np.nan

k1=0
k2=0

for j in range(-10,10):
    for i in range(0, len(df)):
        if df['catdist_fron'][i]==j:
            RH_macS[k2,:]=np.array(df_all['q SMAC'][i])
            RH_macL[k2,:]=np.array(df_all['q LMAC'][i])
            RH_era[k2,:]=np.array(df_all['q ERA'][i])

            k2=k2+1
        MG_macS[k1,:]=np.nanmean(RH_macS, axis=0)
        MG_macL[k1,:]=np.nanmean(RH_macL, axis=0)
        MG_era[k1,:]=np.nanmean(RH_era, axis=0)

    k1=k1+1
    k2=0



#*****************************************************************************\
# Plot one example pre and post


for i in range(13,20): #Entre 3 y 10 deg post cold front
#************************#****************************************************\
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
#************************#****************************************************\
#Pressure
    ax1.plot(MG_macL[i,:],pres_ei,'r-o', label='MAC lin_int')
    ax1.plot(MG_macS[i,:],pres_ei,'g-o', label='MAC spl_int')
    ax1.plot(MG_era[i,:],pres_ei,'y-o', label='ERAi')

    ax1.set_ylim(1000,600)
    ax1.set_yticks(np.arange(600, 1050, 50))
    ax1.set_xlim(0,6)
    ax1.set_title('Post Cold Front - ' +str(i-9) +' deg', size=12)
    ax1.set_ylabel('Pressure (hPa)',fontsize = 10)
    ax1.set_xlabel('q (g kg$^{-1}$)',fontsize = 10)
    ax1.legend(loc=1,fontsize = 10)
    ax1.grid()

    #Altitude
    ax2.plot(MG_macL[i,:],hght_ei,'r-o', label='MAC lin_int')
    ax2.plot(MG_macS[i,:],hght_ei,'g-o', label='MAC spl_int')
    ax2.plot(MG_era[i,:],hght_ei,'y-o', label='ERAi')

    ax2.set_ylim(0,5000)
    ax2.set_xlim(0,6)
    ax2.set_ylabel('Altitude (m)',fontsize = 10)
    ax2.set_xlabel('q (g kg$^{-1}$)',fontsize = 10)
    ax2.grid()

    plt.savefig(path_data_save + 'mean_postcoldfront_'+ str(i-9)  + 'deg.eps', format='eps', dpi=300)

#************************#****************************************************\
# Pre
#************************#****************************************************\

for i in range(3,7): #Entre 3 y 10 deg post cold front
#************************#****************************************************\
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
#************************#****************************************************\
#Pressure
    ax1.plot(MG_macL[i,:],pres_ei,'r-o', label='MAC lin_int')
    ax1.plot(MG_macS[i,:],pres_ei,'g-o', label='MAC spl_int')
    ax1.plot(MG_era[i,:],pres_ei,'y-o', label='ERAi')

    ax1.set_ylim(1000,600)
    ax1.set_yticks(np.arange(600, 1050, 50))
    ax1.set_xlim(0,6)
    ax1.set_title('Post Pre Front - ' +str(i) +' deg', size=12)
    ax1.set_ylabel('Pressure (hPa)',fontsize = 10)
    ax1.set_xlabel('q (g kg$^{-1}$)',fontsize = 10)
    ax1.legend(loc=1,fontsize = 10)
    ax1.grid()

    #Altitude
    ax2.plot(MG_macL[i,:],hght_ei,'r-o', label='MAC lin_int')
    ax2.plot(MG_macS[i,:],hght_ei,'g-o', label='MAC spl_int')
    ax2.plot(MG_era[i,:],hght_ei,'y-o', label='ERAi')

    ax2.set_ylim(0,5000)
    ax2.set_xlim(0,6)
    ax2.set_ylabel('Altitude (m)',fontsize = 10)
    ax2.set_xlabel('q (g kg$^{-1}$)',fontsize = 10)
    ax2.grid()

    plt.savefig(path_data_save + 'mean_precoldfront_'+ str(i+1)  + 'deg.eps', format='eps', dpi=300)


