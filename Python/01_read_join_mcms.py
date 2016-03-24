import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#import pandas as pd
from glob import glob

#******************************************************************************
#Lee archivos MCMS
#******************************************************************************



#cyc=np.loadtxt(fname='/home/flang/Dropbox/Monas Uni/SO/Mac Island Soundings/Read_Files/MCMS/mcms_erai_2006_tracks.txt', delimiter=' ')
cyc=np.loadtxt(fname='./mcms_erai_2006_tracks.txt', delimiter=' ')
#cyc=open('/home/flang/Dropbox/Monas Uni/SO/Mac Island Soundings/Read_Files/MCMS/mcms_erai_2006_tracks.txt')


CoLat=cyc[:,6]
lat = 90.0 - (CoLat*0.01);

#**************************************************************************
#Extrar SH
#**************************************************************************
#elimina los negativos
indsr = np.nonzero(lat>0)
lat2=lat[indsr] 

#Filtra los archivos
cyc[indsr,:]

#******************************************************************************
#Fecha
#******************************************************************************
array=cyc[:,0:4] #Array fechas

yy=array[:,0].astype(int) 
mm=array[:,1].astype(int) 
dd=array[:,2].astype(int)
hh=array[:,3].astype(int) 	 	

ndates,_=array.shape

#print ndates
mydates=np.array([])
for n in range(0,ndates):
	mydates=np.append(mydates, datetime(yy[n], mm[n], dd[n],hh[n],0,0))

