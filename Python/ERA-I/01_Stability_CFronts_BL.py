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
from scipy import array, arange, exp
from scipy.interpolate import interp1d
from pylab import plot,show, grid, xlabel, ylabel, xlim, ylim, yticks, legend
from matplotlib import gridspec
from numpy import inf


base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/Stability/'


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
q_erai=matb1['q'][:]#kg/kg
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

    temp_erai=np.concatenate((temp_erai,temp_r), axis=0)
    rh_erai=np.concatenate((rh_erai,rh_r), axis=0)
    q_erai=np.concatenate((q_erai,q_r), axis=0)
    v_erai=np.concatenate((v_erai,v_r), axis=0)
    u_erai=np.concatenate((u_erai,u_r), axis=0)
    time_erai=np.concatenate((time_erai,time_r), axis=1)


#*****************************************************************************\
#Height Levels
file_levels = np.genfromtxt('./levels.csv', delimiter=',')
hght_ei=file_levels[:,2]*1000 #meters #*****************************************************************************\
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


rh_ei[rh_ei>100]=100
rh_ei[rh_ei<0]=0
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
pot_temp_ei=np.empty(theta_ei.shape)*np.nan

e_ei=np.empty(theta_ei.shape)*np.nan
es_ei=np.empty(theta_ei.shape)*np.nan
dewp_ei=np.empty(theta_ei.shape)*np.nan

#es=6.112 * np.exp((17.67 * (temp_ei-273.16))/float((temp_ei-273.16) + 243.5))

#Potential Temperature and Dew point
for j in range(0,len(time)):
    for i in range(0,len(pres_ei)):
        pot_temp_ei[j,i]=(temp_ei[j,i])*((1000./pres_ei[i])**0.286)

        es_ei[j,i]=6.112 * np.exp((17.67 * (temp_ei[j,i]-273.16))/float((temp_ei[j,i]-273.16) + 243.5))
        e_ei[j,i] = es_ei[j,i] * (rh_ei[j,i]/float(100))
        dewp_ei[j,i] = np.log(e_ei[j,i]/float(6.112))*243.5/float(17.67-np.log(e_ei[j,i]/float(6.112)))+273.16

#Potential Virtual Temperature Gradient and Wind Shear
for j in range(0,len(time)):
    for i in range(1,len(pres_ei)-1):
        dthetav_ei[j,i]=(thetav_ei[j,i+1]-thetav_ei[j,i])/float(hght_ei[i+1]-hght_ei[i])

        vshear_ei[j,i]=np.sqrt(float((u_ei[j,i]-u_ei[j,i-1])**2+(v_ei[j,i]-v_ei[j,i-1])**2))/float(hght_ei[i]-hght_ei[i-1])

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

#*****************************************************************************\

for j in range(0,ni[1]):
#Calculate new variables
    for i in range(0,len(hght_ei)):

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
dewp_pres[dewp_pres== inf] = np.nan
dewp_pres[dewp_pres== -inf] = np.nan



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
hght_my=hght_my_exp.T

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            Stability
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#MAC
#*****************************************************************************\
EIS_my=np.empty(ni[1])
EIS_my[:]=np.nan

LTS_my=np.empty(ni[1])
LTS_my[:]=np.nan
LCL_my=np.empty(ni[1])
LCL_my[:]=np.nan
T_gamma=np.empty(ni[1])
T_gamma[:]=np.nan
Lv=np.empty(ni[1])
Lv[:]=np.nan
qs=np.empty(ni[1])
qs[:]=np.nan

gamma_my=np.empty(ni[1])
gamma_my[:]=np.nan

z700=np.empty(ni[1])
z700[:]=np.nan

p1=np.empty(ni[1])
p1[:]=np.nan
p2=np.empty(ni[1])
p2[:]=np.nan

Tcel=np.empty(ni[1])
Tcel[:]=np.nan
es=np.empty(ni[1])
es[:]=np.nan


#Constante
cp=1005.7 #J/(kg K)
Rd=287.04 #J/(kg K)
Rv=461.5 #J/(kg K)
g=9.8 #m/s2
eps = Rd/float(Rv)

for i in range(0,ni[1]):
# Lower-tropospheric Stability Parameter
    LTS_my[i]=pot_temp_my[i,11]-pot_temp_my[i,0] #pres_ei[11]=700 hPa
# Lifting condensation level
    LCL_my[i]=125*(temp_my[i,0]-dewp_my[i,0])
# Variables to calculate Moist-adiabatic potential temperature gradient at 850
    z700[i]=(Rd*temp_my[i,0])/float(g)*np.log(pres_ei[0]/float(700)) #Temp in K
    #z700[i]=hght_ei[11]

    T_gamma[i]=(pot_temp_my[i,0]+pot_temp_my[i,11])/2.
    #qs[i]=100*mixr[i,6]/float(relh[i,6]) #at 850
    Lv[i]=(2.501-0.00237*(temp_my[i,6]-273.16))*10**6 # J/kg

#Saturation specific humidity
    Tcel[i]=temp_my[i,6]-273.16 #C
    es[i] = 6.112 * np.exp(17.67*Tcel[i]/float(Tcel[i]+243.5)) #vapor pressure in mb
    qs[i] = eps * es[i] / float(850 - (1 - eps) * es[i] )*1000 #g/kg

# Moist-adiabatic potential temperature gradient at 850

    p1[i]=1+((Lv[i]*qs[i])/float(Rd*T_gamma[i]))
    p2[i]=1+(((Lv[i]**2)*qs[i])/float(cp*Rv*(T_gamma[i]**2)))

    gamma_my[i]=(g/float(cp))*(1-(p1[i]/float(p2[i])))

# Estimated Inversion Strenght
    EIS_my[i]=LTS_my[i]-(gamma_my[i]*(z700[i]-LCL_my[i]))


#*****************************************************************************\
# ERAi
#*****************************************************************************\
EIS_ei=np.empty(ni[1])
EIS_ei[:]=np.nan

LTS_ei=np.empty(ni[1])
LTS_ei[:]=np.nan
LCL_ei=np.empty(ni[1])
LCL_ei[:]=np.nan
T_gamma=np.empty(ni[1])
T_gamma[:]=np.nan
Lv=np.empty(ni[1])
Lv[:]=np.nan
qs=np.empty(ni[1])
qs[:]=np.nan

gamma_ei=np.empty(ni[1])
gamma_ei[:]=np.nan

z700=np.empty(ni[1])
z700[:]=np.nan

p1=np.empty(ni[1])
p1[:]=np.nan
p2=np.empty(ni[1])
p2[:]=np.nan

Tcel=np.empty(ni[1])
Tcel[:]=np.nan
es=np.empty(ni[1])
es[:]=np.nan


#Constante
cp=1005.7 #J/(kg K)
Rd=287.04 #J/(kg K)
Rv=461.5 #J/(kg K)
g=9.8 #m/s2
eps = Rd/float(Rv)

for i in range(0,ni[1]):
# Lower-tropospheric Stability Parameter
    LTS_ei[i]=pot_temp_ei[i,11]-pot_temp_ei[i,0] #pres_ei[11]=700 hPa
# Lifting condensation level
    LCL_ei[i]=125*(temp_ei[i,0]-dewp_ei[i,0])
# Variables to calculate Moist-adiabatic potential temperature gradient at 850
    z700[i]=(Rd*temp_ei[i,0])/float(g)*np.log(pres_ei[0]/float(700)) #Temp in K
    #z700[i]=hght_ei[11]

    T_gamma[i]=(pot_temp_ei[i,0]+pot_temp_ei[i,11])/2.
    #qs[i]=100*mixr[i,6]/float(relh[i,6]) #at 850
    Lv[i]=(2.501-0.00237*(temp_ei[i,6]-273.16))*10**6 # J/kg

#Saturation specific humidity
    Tcel[i]=temp_ei[i,6]-273.16 #C
    es[i] = 6.112 * np.exp(17.67*Tcel[i]/float(Tcel[i]+243.5)) #vapor pressure in mb
    qs[i] = eps * es[i] / float(850 - (1 - eps) * es[i] )*1000 #g/kg

# Moist-adiabatic potential temperature gradient at 850

    p1[i]=1+((Lv[i]*qs[i])/float(Rd*T_gamma[i]))
    p2[i]=1+(((Lv[i]**2)*qs[i])/float(cp*Rv*(T_gamma[i]**2)))

    gamma_ei[i]=(g/float(cp))*(1-(p1[i]/float(p2[i])))

# Estimated Inversion Strenght
    EIS_ei[i]=LTS_ei[i]-(gamma_ei[i]*(z700[i]-LCL_ei[i]))




nfin=1500
plot(LTS_ei[0:nfin],'b')
plot(EIS_ei[0:nfin],'r')
plot(LTS_my[0:nfin],'c')
plot(EIS_my[0:nfin],'m')
grid()
show()


