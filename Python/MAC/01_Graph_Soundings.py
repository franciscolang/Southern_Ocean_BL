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
path_data_save=base_dir+'/Dropbox/Monash_Uni/Conferences/2017 AMOS/figures/'

latMac=-54.50;
lonMac=158.95;

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
matb1= sio.loadmat(path_databom+'BOM_2009.mat')
bom_in=matb1['BOM_S'][:]
timesd= matb1['time'][:]
bom=bom_in

# for y in range(2007,2011):
#     matb= sio.loadmat(path_databom+'BOM_'+str(y)+'.mat')
#     bom_r=matb['BOM_S'][:]
#     timesd_r= matb['time'][:]
#     bom=np.concatenate((bom,bom_r), axis=2)
#     timesd=np.concatenate((timesd,timesd_r), axis=1)

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

timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
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
#*****************************************************************************\
#*****************************************************************************\
#                               Plot 1
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\


# ncase=614 #(2008, 11, 10, 23, 0))
ncase=509 #(2009, 9, 17, 11, 0)

#*****************************************************************************\
#Original
#*****************************************************************************\

RH=relh[:,ncase]
WSP=wspd[:,ncase]
DIR=wdir[:,ncase]
T=temp[:,ncase]
DP=dewp[:,ncase]
H=hght[:,ncase]
date=time_my[ncase]
P=pres[:,ncase]

#*****************************************************************************\
height=H
pressure=P
temperature=T
dewpoint=DP
wsp=WSP*1.943844
wdir=DIR

mydataM1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ', 'MAC ' +str(date))))

S=SkewT.Sounding(soundingdata=mydataM1)
S.plot_skewt(color='r')

plt.savefig(path_data_save + 'sounding.png', format='png', dpi=600)
#*****************************************************************************\
#Original
#*****************************************************************************\

RH2=relhum_my[ncase,:]
WSP2=wsp_my[ncase,:]
DIR2=wdir_my[ncase,:]
T2=temp_my[ncase,:]
DP2=dewp_my[ncase,:]
H2=hght_my[ncase,:]
date=time_my[ncase]
P2=pres_ei

T2[0]=DP2[0]
DP2[0]=temp_my[ncase,0]


#*****************************************************************************\
height=H2
pressure=P2
temperature=T2
dewpoint=DP2
wsp=WSP2*1.943844
wdir=DIR2

mydataM2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height,pressure,temperature,dewpoint,wdir,wsp,' ', 'MAC ' +str(date))))

S=SkewT.Sounding(soundingdata=mydataM2)
S.plot_skewt(color='r')

show()
# S.make_skewt_axes()
# S.add_profile(color='r',bloc=0)


#plt.savefig(path_data_save + '5kC1_C'+str(i+1)+'.png', format='png', dpi=1200)



