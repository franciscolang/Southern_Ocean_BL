import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#import pandas as pd
from glob import glob

#******************************************************************************
#Leer Multiples Archivos MCMS
#******************************************************************************

#fnames = glob('mcms_erai_2009_tracks.txt')
fnames = glob('./files_ori/*.txt')
arrays = [np.loadtxt(f, delimiter=' ') for f in fnames]
array_cyc = np.concatenate(arrays, axis=0)
print len(array_cyc)

#np.savetxt('result.txt', final_array, fmt='%.2f')

array=array_cyc[:,0:4] #Array fechas

yy=array[:,0].astype(int) 
mm=array[:,1].astype(int) 
dd=array[:,2].astype(int)
hh=array[:,3].astype(int) 	 	

ndates,_=array.shape

#print ndates
mydates=np.array([])
for n in range(0,ndates):
	mydates=np.append(mydates, datetime(yy[n], mm[n], dd[n],hh[n],0,0))
	#mydates=np.append(mydates, datetime(yy[n], 1, 1,0,0,0))

print mydates