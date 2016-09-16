import numpy as np
import scipy.io as sio
import os
from skewt import SkewT
from matplotlib.pyplot import rcParams,figure,show,draw
from datetime import datetime, timedelta

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
caso=297 #1999-05-31 11:00:00
#caso=295 #1999-05-30 11:00:00
#caso=114 #2006-02-26 23:00:00
#print time_my[caso]

n_nnan=np.count_nonzero(~np.isnan(pres[:,caso]))

#*****************************************************************************\
height_m=hght[0:n_nnan-1,caso]
pressure_pa=pres[0:n_nnan-1,caso]
temperature_c=temp[0:n_nnan-1,caso]
dewpoint_c=dwpo[0:n_nnan-1,caso]
wsp=wspd[0:n_nnan-1,caso]
wdir=wdir[0:n_nnan-1,caso]


print n_nnan

mydata=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'BoM',time_my[caso])))

S=SkewT.Sounding(soundingdata=mydata)
S.plot_skewt(color='r')
show()

