import numpy as np
import scipy.io as sio
import os
from pylab import plot,show, grid
import math
import pandas as pd

base_dir = os.path.expanduser('~')
latMac=-54.50;
lonMac=158.95;

#mask = np.all(np.isnan(Clat), axis=0)
#Clat[~mask,:]

#*****************************************************************************\
#*****************************************************************************\
#                                   Fronts
#*****************************************************************************\
#*****************************************************************************\
#Reading
path_front=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_fronts/'
#cada col es un frente, cada row es una posicion del frente

# matb1= sio.loadmat(path_front+'FRONTS_2006.mat')
# matb2= sio.loadmat(path_front+'FRONTS_2007.mat')
# matb3= sio.loadmat(path_front+'FRONTS_2008.mat')
# matb4= sio.loadmat(path_front+'FRONTS_2009.mat')
# matb5= sio.loadmat(path_front+'FRONTS_2010.mat')
# cf06=matb1['cold_fronts'][:] #(2,100 row ,200 col ,1460)
# cf07=matb2['cold_fronts'][:]
# cf08=matb3['cold_fronts'][:]
# cf09=matb4['cold_fronts'][:]
# cf10=matb5['cold_fronts'][:]
# # # c = np.concatenate([aux[..., np.newaxis] for aux in sequence_of_arrays], axis=3)
# cold_fronts = np.concatenate((cf06,cf07,cf08,cf09,cf10), axis=3)
#cold_fronts = np.concatenate((cf06,cf07,cf08), axis=3)
#cold_fronts =cf06



matb1= sio.loadmat(path_front+'FRONTS_1995.mat')
cf_in=matb1['cold_fronts'][:]
cf=cf_in

for y in range(1996,2011):
    matb= sio.loadmat(path_front+'FRONTS_'+str(y)+'.mat')
    cf_r=matb['cold_fronts'][:]
    cf=np.concatenate((cf,cf_r), axis=3)


cold_fronts=cf





print cold_fronts.shape
#*****************************************************************************\
n=cold_fronts.shape[3]
#n=4384
#n=1460

Clat=np.array([100,200,n])
Clon=np.array([100,200,n])

Clon=cold_fronts[1,:,:,:]
Clat=cold_fronts[0,:,:,:]

Clat[Clat==-9999]=np.nan
Clon[Clon==-9999]=np.nan

#*****************************************************************************\
#Define new variables
DCfront=np.empty([200,n])*np.nan
CCfront=np.empty([n])*np.nan
#*****************************************************************************\
#Calculate Distance to MAC
#*****************************************************************************\
# i=33
# j=294
# x=Clon[:,i,j]
# y=Clat[:,i,j]
for j in range(0,n-1):
    for i in range(0,199):
        x=Clon[:,i,j]
        y=Clat[:,i,j]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        x0=lonMac
        y0=latMac
        if len(x)>1 and np.count_nonzero(x)>1 and np.count_nonzero(y)>1: #At least two points
            #******************************************************************
            #Line equation
            m,b=np.polyfit(x,y,1)
            recta=m*x+b

            x2=x[0]
            x1=x[-1]
            y2=recta[0]
            y1=recta[-1]
            #y2=-52
            #******************************************************************
            #Calculate Distance MAC to front line
            px = x2-x1
            py = y2-y1
            something = px*px + py*py
            u =  ((x0 - x1) * px + (y0 - y1) * py) / float(something)

            if u > 1: #Revisar para dar un rango de error fuera
                u = np.nan
                dist=np.nan
            elif u < 0:
                u = np.nan
                dist=np.nan
            else:
                xx = x1 + u * px
                yy = y1 + u * py

                dx = xx - x0
                dy = yy - y0
                dist = math.sqrt(dx*dx + dy*dy)

            #Asigna de acuerdo si es antes o despues del frente.
                if (dx<0. and dy<0.): #before ok
                    dist=dist*(-1)
                elif (dx>0. and dy>0.):#after ok
                    dist=dist
                elif (dx<0. and dy>0.):#before ok
                    dist=dist*(-1)
                elif (dx>0. and dy<0.):#after ok
                    dist=dist
                # print dist
                # plot(x,y,'yo',x,recta,'--k',x1,y1,'go',x2,y2,'ro',x0,y0,'bo')
                # grid()
                # show()
            #******************************************************************
        else:
            dist=np.nan

    #Construir nueva matriz (1,200,1460)
        DCfront[i,j]=np.array(dist)

#*****************************************************************************\
#Calculate closest front to MAC
#*****************************************************************************\

for j in range(0,n-1):
    if np.all(np.isnan(DCfront[:,j])): #read column if all are nan
        CCfront[j] = np.nan
    else:
        idx = np.nanargmin(np.abs(DCfront[:,j])) #position closest to zero
        CCfront[j] = DCfront[idx,j]

CCfront[np.nonzero(abs(CCfront)>15)]=np.nan
#no_nan=np.count_nonzero(~np.isnan(CCfront))
#*****************************************************************************\
#Creating Dataframe Pandas
#*****************************************************************************\
date_fronts = pd.date_range('1995-01-01 00:00', periods=n, freq='6H')
dff={'Dist C Front':CCfront}

df_front1 = pd.DataFrame(data=dff,index=date_fronts)
df_front1.index.name = 'Date'

#Every 12 hours
date_index_12 = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
df_front=df_front1.reindex(date_index_12)
#Number of cases
no_nan=np.count_nonzero(~np.isnan(df_front['Dist C Front']))
print no_nan
#*****************************************************************************\
#Saving CSV
#*****************************************************************************\
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

#df_front.to_csv(path_data_save + 'df_front.csv', sep='\t', encoding='utf-8')
df_front.to_csv(path_data_save + 'df_cfront_19952010.csv', sep='\t', encoding='utf-8')
