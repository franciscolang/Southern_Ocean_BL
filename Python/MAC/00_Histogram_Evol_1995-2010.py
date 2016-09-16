import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/MatFiles/files_bom/Sensitivity/'
#*****************************************************************************\
matb1= sio.loadmat(path_data+'BOM_Evolution_1995-2010.mat')
matb2= sio.loadmat(path_data+'BOM_Levels_1995-2010.mat')

nlevsd=matb2['nlevssd'][:] #Numero de niveles entre 0 y 2500
nsd=matb2['nsd'][:] #numero de soundings per year
perc_clas=matb1['Y'][:] #'No Inv.' ,'Single Inv.', 'Decoupled L.', 'Buffer L.'
anio=np.arange(1995,2011)
#*****************************************************************************\
#Plotting
#*****************************************************************************\
# # Setting Plot
# fig, ax0 = plt.subplots(figsize=(10, 5))

# width = 0.2
# pos = list(range(len(anio)))
# #*****************************************************************************\
# ax0.bar(pos,
#         #using df['pre_score'] data,
#         perc_clas[:,0],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#c6c9d0',
#         #color='#7C83AF',
#         # with label the first value in first_name
#         label='No Inv.')


# ax0.bar([p + width for p in pos],
#         #using df['mid_score'] data,
#         perc_clas[:,1],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#67832F',
#         # with label the second value in first_name
#         label='Single Inv.')

# ax0.bar([p + width*2 for p in pos],
#         #using df['post_score'] data,
#         perc_clas[:,2],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='tomato',
#         #color='#182157',
#         # with label the third value in first_name
#         label='Decoupled L.')


# ax0.bar([p + width*3 for p in pos],
#         #using df['post_score'] data,
#         perc_clas[:,3],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='blue',
#         #color='#080F3A',
#         # with label the third value in first_name
#         label='Buffer L.')

# # Set the position of the x ticks
# ax0.set_xticks([p + 2 * width for p in pos])
# # Set the labels for the x ticks
# ax0.set_xticklabels(anio,fontsize = 11)
# # Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*5)
# ax0.set_ylabel('Percentage')
# #Legend
# plt.legend(['No Inv.', 'Single', 'Decoupled L.', 'Buffer L.'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 11)
# plt.grid()
# plt.show()



# #*****************************************************************************\
# # Setting Plot
# fig, ax1 = plt.subplots(figsize=(10, 4))
# # Set the bar width
# bar_width = 0.5

# # positions of the left bar-boundaries
# bar_l = [i+1 for i in range(len(anio))]

# # positions of the x-axis ticks (center of the bars as bar labels)
# tick_pos = [i+(bar_width/2) for i in bar_l]

# # Create the plot bars in x position
# ax1.bar(bar_l,
#         # using the pre_score data
#         nlevsd,
#         # set the width
#         width=bar_width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='orange')

# # set the x ticks with names
# plt.xticks(tick_pos, anio,fontsize = 11)
# # add a grid
# plt.grid()
# # set axes labels and title
# ax1.set_ylabel('Number of levels below 2500 mts.')
# #plt.title('Mean Scores For Each Test')
# # Set a buffer around the edge
# plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

# plt.show()
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

# # Setting Plot
fig, ax0 = plt.subplots(figsize=(5.5, 10))

width = 0.2
pos = list(range(len(anio)))

ax0.barh(pos,
        #using df['pre_score'] data,
        perc_clas[:,0],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='No Inv.')


ax0.barh([p + width for p in pos],
        #using df['mid_score'] data,
        perc_clas[:,1],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='Single Inv.')

ax0.barh([p + width*2 for p in pos],
        #using df['post_score'] data,
        perc_clas[:,2],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='Decoupled L.')


ax0.barh([p + width*3 for p in pos],
        #using df['post_score'] data,
        perc_clas[:,3],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#080F3A',
        # with label the third value in first_name
        label='Buffer L.')

# Set the position of the x ticks
ax0.set_yticks([p + 2 * width for p in pos])
# Set the labels for the x ticks
ax0.set_yticklabels(anio,fontsize = 13)
# Setting the x-axis and y-axis limits
plt.ylim(min(pos)-width, max(pos)+width*5)
ax0.set_xlabel('Percentage')
#Legend
plt.legend(['No Inv.', 'Sin. Inv.', 'Dec. L.', 'Buffer L.'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 11)
plt.grid()
plt.show()



#*****************************************************************************\
# Setting Plot
fig, ax1 = plt.subplots(figsize=(5.5, 10))
# Set the bar width
bar_width = 0.5
width=0.5
# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(anio))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]

# Create the plot bars in x position
ax1.barh(bar_l,
        # using the pre_score data
        nlevsd,
        # set the width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='orange')

# set the x ticks with names
plt.yticks(tick_pos, anio,fontsize = 13)
# add a grid
plt.grid()
# set axes labels and title
ax1.set_xlabel('Number of levels below 2500 mts.')
#plt.title('Mean Scores For Each Test')
# Set a buffer around the edge
plt.ylim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.show()
