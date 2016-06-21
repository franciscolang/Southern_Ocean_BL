import numpy as np
import pandas as pd
from datetime import datetime

#Date Range
#******************************************************************************
#date_range = pd.date_range('2015-01-01 00:00', periods=60, freq='6H')
#date_range = pd.data_range('2015-01-01 00:00', '2015-01-01 00:59', freq='1T')
date_range = pd.date_range('2015-01-01 00:00', periods=60, freq='1T')

print date_range[0]
#print date_range[1]
#print date_range[2]
#print date_range[5]

#Data Frame
#******************************************************************************
 
data = np.arange(1,61)
df = pd.DataFrame(data,index=date_range)

#A DataFrame filled with NaNs:
cols=['col1','col2','col3']
nandf = pd.DataFrame(index=date_range, columns=cols)

data1=np.random.rand(60)
data2=data1*10
data3=data1*100

#Slicing
d={'col1':data1, 'col2':data2, 'col3':data3}
df=pd.DataFrame(data=d,index=date_range)
print df 
df['col1'][2:4] #Indica columna y filas para extraer
df.ix[2:4, 0] #igual q antes pero el 0 indica la columna


df.iloc[[3]] # retrieve single value
df.iloc[2:4] # retrieve a series

foo1=datetime(2015,1,1,0,20)
foo2=datetime(2015,1,1,0,30)
print df.loc[foo1:foo2]

df_idx_sparse = df.loc[foo1:foo2].index
print df.loc[df_idx_sparse]

#np.savetxt(r'./prueba.txt', df.values, fmt='%10.5f')
df.to_csv('./prueba_csv', date_format='%Y-%m-%d %H:%M:%S', float_format='%9.6f', header=None, sep='\t')

#pd.read_csv("whitespace.csv", header=None, delimiter=r"\s+")