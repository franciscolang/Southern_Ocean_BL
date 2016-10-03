import numpy as np
import scipy.io as sio
import os
from skewt import SkewT
from matplotlib.pyplot import rcParams,figure,show,draw, subplot
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

base_dir = os.path.expanduser('~')
#*****************************************************************************\
path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
matb1= sio.loadmat(path_databom+'BOM_1999.mat')
bom=matb1['BOM_S'][:]
timesd= matb1['time'][:]
timesd=timesd[0,:]

ni=bom.shape
nd=ni[2]
#*****************************************************************************\
pres=bom[:,0,:].reshape(ni[0],nd)
hght=bom[:,1,:].reshape(ni[0],nd)
temp=bom[:,2,:].reshape(ni[0],nd)
mixr=bom[:,5,:].reshape(ni[0],nd)
wdir=bom[:,6,:].reshape(ni[0],nd)
wspd=bom[:,7,:].reshape(ni[0],nd)
relh=bom[:,4,:].reshape(ni[0],nd)
dwpo=bom[:,3,:].reshape(ni[0],nd)
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
#Cambiar fechas
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
#*****************************************************************************\
#Seleccionando Caso
#caso=297 #1999-05-31 11:00:00
caso=295 #1999-05-30 11:00:00
#caso=114 #2006-02-26 23:00:00
#print time_my[caso]

n_nnan=np.count_nonzero(~np.isnan(pres[:,caso]))

#*****************************************************************************\
height_m=hght[0:n_nnan-1,caso]
pressure_pa=pres[0:n_nnan-1,caso]
temperature_c=temp[0:n_nnan-1,caso]
dewpoint_c=dwpo[0:n_nnan-1,caso]
wsp_kn=wspd[0:n_nnan-1,caso]
wdir_deg=wdir[0:n_nnan-1,caso]

mydata1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir_deg,wsp_kn,'BoM',time_my[caso])))


#*****************************************************************************\
# S=SkewT.Sounding(soundingdata=mydata1)
# S.plot_skewt(color='r')


#*****************************************************************************\
# SkewT
#*****************************************************************************\
from matplotlib.projections import register_projection
from skewx_projection_matplotlib_lt_1d4 import SkewXAxes
register_projection(SkewXAxes)

T=temperature_c
p=pressure_pa
Td=dewpoint_c

from matplotlib import gridspec


fig=plt.figure(figsize=(8, 6))
#gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) #fila, col
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
ax0=fig.add_subplot(gs[0], projection='skewx')
plt.grid(True)

ax0.semilogy(T, p, 'r')
ax0.semilogy(Td, p, 'g')

l = ax0.axvline(0, color='b')

ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.set_yticks(np.linspace(100, 1000, 10))
ax0.set_ylim(1050, 100)

ax0.xaxis.set_major_locator(MultipleLocator(10))
ax0.set_xlim(-40, 30)

ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
ax0.set_xlabel('Temperature (C)',fontsize = 10)



#*****************************************************************************\
# Components
#*****************************************************************************\
wspd=wsp_kn*0.54444444

u=wspd*(np.cos(np.radians(270-wdir_deg)))
v=wspd*(np.sin(np.radians(270-wdir_deg)))

ax1=fig.add_subplot(gs[1])
ax1.semilogy(u,pressure_pa,label='u')
ax1.semilogy(v,pressure_pa,label='v', color='r')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.axvline(0, color='black')
ax1.set_xlim(-60,60)
ax1.legend(loc='upper left',fontsize = 10,ncol=1)
ax1.set_xlabel('(ms$^{-1}$)',fontsize = 10)
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.set_yticks(np.linspace(100, 1000, 10))
ax1.set_ylim(1050, 100)
ax1.grid()
ax1.axes.yaxis.set_ticklabels([])
plt.tight_layout()
plt.show()
