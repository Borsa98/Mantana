import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time
from datetime import date
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from matplotlib import animation
path = os.path.dirname(os.getcwd())
print('\n\t'+path)

# Script for Spatial Interpolation and Cross Validation with Kriging
goal = 'Interpolation' # Interpolation
variable = 'Precipitation' # Temperature
# Number of stations with which you want to predict the unknown stations (8, 16 and 32)
nstationbasis = 8

# --------------------------------------------LOAD THE DATA-----------------------------------

dfstats  = pd.read_csv(r'C:/Users/amede/Downloads/Practical Session 3 - Kriging - Variogram-20230513/P_ADIGE.in',sep=' ',skiprows=3, header=None)
dates = pd.date_range(start='12/29/1955',end='12/31/2013')
dfstats.insert(0, 'index', dates)
dfstats = dfstats.replace(-9999,np.nan)
# definiton of the coordinates df 
coord_df = dfstats[:3]
coord_df.iloc[0,0], coord_df.iloc[1,0], coord_df.iloc[2,0] ='X','Y','Z'
# define the dfstats
dfstats = dfstats[3:]
#mask = (dfstats['index'] >= '1988-2-1') & (dfstats['index'] <= '1990-2-11')
#mask = (dfstats['index'] >= '1986-8-17') & (dfstats['index'] <= '1987-8-26')
mask = (dfstats['index'] >= '1986-8-17') & (dfstats['index'] <= '2013-12-31')
dfstats=dfstats.loc[mask]
#╬dfstats = dfstats.loc[dfstats['index']>='01/01/2000']
# recreate the global df
#dfstats1=coord_df.append(dfstats)
dfstats = pd.concat([coord_df,dfstats],axis=0)

numboriginalstats = dfstats.shape[1]
dfstats = dfstats.set_index('index')
coordprec = dfstats.copy()

# -------------------------------------------------------------------------------------------
if goal == 'Interpolation':
    dfcentroids = pd.read_csv(r"C:\Users\amede\Documents\mancentroids.csv")#put centroids file here
    dfcentroids.dropna(axis=0,inplace=True, how='any')
    dfcentroids.rename(columns={'Xcoord':'X', 'Ycoord':'Y','SAMPLE_Z1': 'Z'}, inplace=True)
    centroids = dfcentroids.T
    df = pd.concat([coordprec,centroids], axis=1, join='outer')
    coordord = df.T.reset_index(drop=True)
    coordprec = coordord.T



# Global distances matrix computation
statglob = pd.Series(coordprec.columns)
xglob = np.array(coordprec.iloc[0])
yglob = np.array(coordprec.iloc[1])
zglob = np.array(coordprec.iloc[2])
coordglob = np.vstack((xglob, yglob)).T

dist_vectglob = pdist(coordglob) # compute the pairwise distance as an array
sqdist_matrixglob = squareform(dist_vectglob) # make the pdist as a squared matrix
# create a df from the square matrix with the name of columns of original df and with the active stations for the day
matrixdfglob = pd.DataFrame(sqdist_matrixglob, columns=coordprec.columns) 


# ------------------------- FILL THIS SECTION with the parameters obtained with gstools' tutorial --------------------------------

#I_a =  6895.417680270454
#var = 35.284710131556686
#var_nug = 7.944844744794626e-32
#change with temperature or precipitation
I_a =  26374.34256212921
var = 30.44025470678838
var_nug = 16.0753602100721

vario =[] 
vario.append(I_a) #I_a
vario.append(var) #var
vario.append(var_nug) #var_nug

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------- KRIGING COMPUTATION MATRIX --------------------------------------------------------------------------------------------------------------

def krig(I_a,var,var_nug,krig_type,z_meas,ztarget,elevstats,matrix_disttouse):
    # I_a = Integral scale; var = variance; var_nug = nugget; krig_type = implemented model: Simple Kriging (SK), Ordinary Kriging (OK), Kriging with External Drift (KED);
    # z_meas = measurements in the known stations; ztarget = elevation of the target station; elevstats = elevations of the known stations; 
    # matrix_disttouse = portion of the matrix of the global distances to use
    
    # Definition of the fundamental variables for the Kriging system
    n_stats = (matrix_disttouse.shape[1] - 1)
    mutualdistances = matrix_disttouse.iloc[1:,1:].values
    disttarget = matrix_disttouse.iloc[1:,0].values

    if krig_type == 'KED':
        dim_mat= n_stats+2
    elif krig_type == 'OK':
        dim_mat= n_stats+1
    elif krig_type == 'SK':
        dim_mat= n_stats
    else:
        print('Kriging type not supported')

    # Definition of matrix and vectors with default values
    matrix = np.ones((dim_mat,dim_mat),dtype=np.float32)
    kt = np.ones(dim_mat,dtype=np.float32) #vector of known terms
    coef_krig = np.zeros(dim_mat,dtype=np.float32) #vector of lambda, unknown terms
    
    # Filling matrix and vectors with the correct values 
    for i in range(n_stats): # it automatically goes till n_stats - 1 (0,1,2...nstats-1)
        matrix[i,i]= var_nug+var # diagonal terms

    for i in range(n_stats):
        for j in range(i+1,n_stats):
            matrix[i,j]=var*np.exp(-mutualdistances[i,j]/I_a)
            matrix[j,i]=matrix[i,j]

    # Vector of the known terms
    for i in range(n_stats):
        r = disttarget[i]/I_a
        kt[i] = var*np.exp(-r)

    # fix the element at position (n+1), (n+1)
    if krig_type in ['KED','OK']:
        matrix[n_stats,n_stats]=0.0    

    # for Kriging with External drift add a line and a column and fill it with the elevation of the stations
    if krig_type == 'KED':
        for i in range(len(elevstats)):
            matrix[i,n_stats+1] = elevstats[i] # elevation
            matrix[n_stats+1,i] = matrix[i,n_stats+1] 
            matrix[n_stats+1,n_stats+1] =0.0
            matrix[n_stats+1,n_stats]=0.0
            matrix[n_stats,n_stats+1]=0.0
        kt[n_stats+1] = ztarget

    #--------------------------- Computation LAMBDA VECTOR ---------------------------------------
    coef_krig = linalg.solve(matrix, kt, overwrite_b = False)
    
    # Computation of the interpolated values 
    if np.all(np.abs(coef_krig[0:n_stats]) > 1.001):
        z_star = np.nan 
        z_var = np.nan

    else:
        if krig_type == 'SK':
            z_meas_sk = z_meas - np.mean(z_meas)
            z_star = np.dot(coef_krig[0:n_stats],z_meas_sk) + np.mean(z_meas)
        else:
            z_star = np.dot(coef_krig[0:n_stats],z_meas)
        
        z_var = var+var_nug-np.dot(coef_krig,kt)    
        
    return (z_star, z_var, matrix)
    # where z_star is the interpolated value, z_var is the variance and matrix is the matrix of the covariance of Kriging
# -------------------------------------------------------------------------------------------------------------------------------------
#  ---------- Set a part the coordinates of the stations (you have already computed the distances in the matrix of distances)-------------------
dfmeas = coordprec.drop(['X','Y','Z'], axis=0)
dfcoord = coordprec.filter(['X','Y','Z'], axis=0)
dfmeas = dfmeas.reset_index() # It's mandatory to obtain ind and not string to do the following loop
dfmeas = dfmeas.drop('index',axis=1) # Just drop also the data column

# Decide the type of Kriging
kriginglist = ['KED']

for kk in kriginglist:
    # Create an empty dataframe for the errors and the interpolated values
    dferrors = pd.DataFrame(np.nan,index=dfmeas.index, columns=dfmeas.columns)
    dfinterpolated = pd.DataFrame(np.nan,index=dfmeas.index, columns=dfmeas.columns)
    #print(kk)
    # Inner Loop for the data period
    for ind in dfmeas.index[0:]:
        print('day number:', ind)
        dfday = dfmeas.iloc[ind].to_frame() #REMEMBER: YOU HAVE TO COMPUTE THE KRIGING DAY BY DAY: THE COVARIANCES ARE THE SAME BUT THE WEIGHTS (UNKNOWN TERMS) CHANGE
        dayrowvect = dfday.T
        #dayrowvect = dfcoord.append(dayrowvect)
        dayrowvect=pd.concat([dfcoord,dayrowvect],axis=0)
        # Indeces of the stations where for the specific day there are measurements (active stations)
        ind_statact = (np.argwhere(~np.isnan(dayrowvect.iloc[3,:].values))).flatten() 
        list_statact = ind_statact.tolist() 
        # Inner Loop for the stations in the basin
        for ic, cols in enumerate(dayrowvect.columns[0:]):
        # for ic in dayrowvect.columns[(numboriginalstats-1):]:
            sortedstat = np.argsort(matrixdfglob.iloc[:,ic]) # indeces of sorted distances between the stats and the stat at the position ic 
            # Intersection between the sorted distanced stats and the act stats (the result will be always the act stats but in the sorted order)
            stat_touse = [x for x in sortedstat if x in list_statact] 
            stat_touse = [ic] + stat_touse[0:nstationbasis] # slice the portion of stations you would like to use
            # Define the variables that have to be called in the function krig for the specific interpolation and cross validation procedure
            matdisttouse = matrixdfglob.iloc[stat_touse,stat_touse] # slice the useful portion of distances matrix
            xtarget,ytarget,elevtarget = dayrowvect.iloc[0,ic],dayrowvect.iloc[1,ic],dayrowvect.iloc[2,ic]
            z_meas = dayrowvect.iloc[3,stat_touse[1:]].to_numpy()  # vector of the known measurements for the stationstouse
            elevstats = dayrowvect.iloc[2,stat_touse[1:]].to_numpy() # vector of the elevations for the stationstouse
            # CALL THE KRIGING FUNCTION
            zstar, z_var, matrix = krig(vario[0],vario[1],vario[2],kk,z_meas,elevtarget,elevstats,matdisttouse) 
            # ---------------------------------- For Precipitation negative values
            if variable == 'Precipitation':
                if zstar<0: 
                    zstar = 0 
            error = np.sqrt((dayrowvect.iloc[3,ic] - zstar)**2) # compute the error and store it in the df (as RMSE)
            dferrors.iloc[ind,ic] = np.array(error)
            dfinterpolated.iloc[ind,ic] = zstar # store the predicted value in the interpolation df
    # ---------------------------------- Rebuild and save the df of the Interpolation values and Errors 
    
    datesnew = pd.date_range(start='8/14/1986',end='12/31/2013')
    #dfcoord=dfcoord.concat(dfcoord.index)
    #dfinterpolated.insert(0, 'index', dates,allow_duplicates=True)
    # dferrors.insert(0, 'index', dates)
    # #dfinterpnew=dfinterpolated
    # #dferrnew=dferrors
    dferrors = dfcoord.append(dferrors)
    #dferrors=pd.concat([dfcoord,dferrors],axis=0)
    dfinterpolated = dfcoord.append(dfinterpolated)
    #dfinterpolated=pd.concat([dfcoord,dfinterpolated],axis=0)
    dfinterpolated.insert(0, 'index', datesnew)
    dferrors.insert(0, 'index', datesnew)
    #  dferrors.iloc[0,0], dferrors.iloc[1,0], dferrors.iloc[2,0] ='X','Y','Z'
    #  dfinterpolated.iloc[0,0], dfinterpolated.iloc[1,0], dfinterpolated.iloc[2,0] ='X','Y','Z'

dferrors.to_csv(r'D:/Kriging/Errors_unknownstats_{}_{}_{}_today.csv'.format(variable,nstationbasis,kk),index=False)
dfinterpolated.to_csv(r'D:/Kriging/Interpolation_{}_{}_{}_today.csv'.format(variable,nstationbasis,kk),index=False)


# dfinterpnew.to_csv(r'D:/Kriging/Errors_unknownstats_{}_{}_{}_todaynew.csv'.format(variable,nstationbasis,kk),index=False)
# dferrnew.to_csv(r'D:/Kriging/Interpolation_{}_{}_{}_todaynew.csv'.format(variable,nstationbasis,kk),index=False)

    #fig=plt.figure()
    
#     def animate(i):
#         print(i)
#         plt.scatter(dfcoord.iloc[0,:],dfcoord.iloc[1,:],c=dfinterpolated.iloc[4+i,:])
    
    
#     anim=animation.FuncAnimation(fig,animate,interval=500)

# anim.save('C:/Users/amede/Downloads/stupidanim.gif',dpi=300, writer=)
