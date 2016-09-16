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
from pylab import plot,show, grid, legend
from skewt import SkewT
from matplotlib.pyplot import rcParams,figure,show,draw

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/Profiles/'
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
#*****************************************************************************\
#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\
#*****************************************************************************\
#925-850-700
df_cluster= pd.read_csv(path_data + 'All_ClusterAnalysis.csv', sep=',', parse_dates=['Date'])

CL4=np.array(df_cluster['Cluster'])
dist_clu=np.array(df_cluster['Distance'])
date=df_cluster['Date']
dates = pd.to_datetime(date)

time_my = dates
for i in range(0,len(dates)):
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
dm={'CL_4':CL4}

df_c = pd.DataFrame(data=dm,index=time_my)
# Eliminate Duplicate Soundings

dfc=df_c.reset_index().drop_duplicates(cols='Date',take_last=True).set_index('Date')

date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
df_clus=dfc.reindex(date_index_all)
df_clus.index.name = 'Date'


#*****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
# Reading MAC
#*****************************************************************************\
# ****************************************************************************\

df_mac= pd.read_csv(path_data_csv + 'df_mac_19952010_5km.csv', sep='\t', parse_dates=['Date'])
df_mac= df_mac.set_index('Date')

df_clumac=pd.concat([df_mac, df_clus],axis=1)

#*****************************************************************************\
#Reading FRONTS
#*****************************************************************************\

df_cfront= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_cfront= df_cfront.set_index('Date')


df_wfront= pd.read_csv(path_data_csv + 'df_wfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_wfront= df_wfront.set_index('Date')

#*****************************************************************************\
#Concadanate dataframes
df_my_wfro=pd.concat([df_clumac, df_wfront],axis=1)
df_my_cfro=pd.concat([df_clumac, df_cfront],axis=1)
#Only Fronts
dfmycfro=df_my_cfro[np.isfinite(df_my_cfro['Dist CFront'])]
dfmywfro=df_my_wfro[np.isfinite(df_my_wfro['Dist WFront'])]
#Both
df_fronts=pd.concat([df_cfront, df_wfront],axis=1)
dfmy_fronts=pd.concat([df_clumac, df_fronts],axis=1)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

df=dfmy_fronts[(np.isfinite(dfmy_fronts['Dist WFront'])) | (np.isfinite(dfmy_fronts['Dist CFront']))] #df.iloc[:,[11,12]]


df_C1 = df[df['CL_4']==1]
df_C2 = df[df['CL_4']==2]
df_C3 = df[df['CL_4']==3]
df_C4 = df[df['CL_4']==4]

hgt_1=np.nanmean(df_C1['1ra Inv'])
hgt_2=np.nanmean(df_C2['1ra Inv'])
hgt_3=np.nanmean(df_C3['1ra Inv'])
hgt_4=np.nanmean(df_C4['1ra Inv'])

stg_1=np.nanmean(df_C1['Strg 1inv'])
stg_2=np.nanmean(df_C2['Strg 1inv'])
stg_3=np.nanmean(df_C3['Strg 1inv'])
stg_4=np.nanmean(df_C4['Strg 1inv'])


C1=np.count_nonzero(~np.isnan(np.array(df_C1['1ra Inv'])))
C2=np.count_nonzero(~np.isnan(np.array(df_C2['1ra Inv'])))
C3=np.count_nonzero(~np.isnan(np.array(df_C3['1ra Inv'])))
C4=np.count_nonzero(~np.isnan(np.array(df_C4['1ra Inv'])))


print C1, C2, C3, C4
print len(df_C1),len(df_C2),len(df_C3),len(df_C4)
