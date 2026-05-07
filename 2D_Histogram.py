import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
import time
import math
from PerodicKDtree_NAMD import PeriodicCKDTree
import matplotlib.cm as cm
import itertools
from MDAnalysis.lib.pkdtree import PeriodicKDTree
import os 
import shutil
import pandas as pd
import argparse
from matplotlib.ticker import FormatStrFormatter, MultipleLocator



''' 
Non-bonded interaction Lennard-Jones formalism with the GenericMOF force field 
Lennard-Jones architecture:	lj_(force field name) = {atom type: [epsilon(K),sigma(A)]}
'''

LJParameters       =   {'O'   	:	[	48.150 	,3.03 ],
                        'N'	    :	[	37.430 	,3.26 ],
                        'C'	    :	[	47.856 	,3.47 ],
                        'F'	    :	[	36.483 	,3.09 ],
                        'B'	    :	[	47.806 	,3.58 ],
                        'P'	    :	[	161.030 ,3.70 ],
                        'S'	    :	[	173.107 ,3.59 ],
                        'W'	    :	[	33.715 	,2.73 ],
                        'V'	    :	[	8.052 	,2.80 ],
                        'I'	    :	[	256.640 ,3.70 ],
                        'U'	    :	[	11.070 	,3.02 ],
                        'K'	    :	[	17.612 	,3.40 ],
                        'Y'	    :	[	36.231 	,2.98 ],
                        'Cl'    :   [	142.562 ,3.52 ],
                        'Br'    :	[	186.191 ,3.52 ],
                        'H'	    :	[	22.142 	,2.57 ],
                        'Zn'	:	[	62.399 	,2.46 ],
                        'Be'	:	[	42.774 	,2.45 ],
                        'Cr'	:	[	7.548 	,2.69 ],
                        'Fe'	:	[	6.542 	,2.59 ],
                        'Mn'	:	[	6.542 	,2.64 ],
                        'Cu'	:	[	2.516 	,3.11 ],
                        'Co'	:	[	7.045 	,2.56 ],
                        'Ga'	:	[	201.288 ,3.91 ],
                        'Ti'	:	[	8.555 	,2.83 ],
                        'Sc'	:	[	9.561 	,2.94 ],
                        'Ni'	:	[	7.548 	,2.52 ],
                        'Zr'	:	[	34.722 	,2.78 ],
                        'Mg'	:	[	55.857 	,2.69 ],
                        'Ne'	:	[	21.135 	,2.89 ],
                        'Ag'	:	[	18.116 	,2.80 ],
                        'In'	:	[	276.771 ,4.09 ],
                        'Cd'	:	[	114.734 ,2.54 ],
                        'Sb'	:	[	276.771 ,3.88 ],
                        'Te'	:	[	286.835 ,3.77 ],
                        'Al'	:	[	155.998 ,3.91 ],
                        'Si'	:	[	155.998 ,3.80 ],
                        'Ca'	:	[	119.766 ,3.03 ],
                        'Nb'	:	[	29.689 	,2.82 ],
                        'Nd'	:	[	5.032 	,3.18 ],
                        'Ge'	:	[	201.288 ,3.80 ],
                        'Sm'	:	[	4.026 	,3.14 ],
                        'Ce'	:	[	6.541 	,3.17 ],
                        'Sn'	:	[	276.771 ,3.98 ],
                        'Au'	:	[	19.626 	,2.93 ],
                        'Ba'	:	[	183.171 ,3.30 ],
                        'Pt'	:	[	40.258 	,2.45 ],
                        'Mo'	:	[	28.180 	,2.72 ],
                        'Sr'	:	[	118.256 ,3.24 ],
                        'Pd'	:	[	24.155 	,2.58 ],
                        'Ru'	:	[	28.180 	,2.64 ],
                        'Pb'	:	[	333.634 ,3.83 ],
                        'Hf'	:	[	36.232 	,2.80 ],
                        'Ho'	:	[	3.523 	,3.04 ],
                        'Eu'	:	[	4.026 	,3.11 ],
                        'Pr'	:	[	5.032 	,3.21 ],
                        'Cs'	:	[	22.645 	,4.02 ],
                        'Na'	:	[	15.097 	,2.66 ],
                        'He'	:	[	10.900 	,2.64 ],
                        'Bi'	:	[	260.660 ,3.89 ],
                        'Li'	:	[	12.580 	,2.18 ],
                        'Se'	:	[	216.380 ,3.59 ],
                        'As'	:	[	206.320 ,3.70 ],
                        'Rb'	:	[	20.128 	,3.67 ],
                        'Tc'	:	[	24.150 	,2.67 ],
                        'Rh'	:	[	26.670 	,2.61 ],
                        'La'	:	[	8.554 	,3.13 ],
                        'Pm'	:	[	4.528 	,3.16 ],
                        'Gd'	:	[	4.529 	,3.00 ],
                        'Tb'	:	[	3.523 	,3.07 ],
                        'Dy'	:	[	3.523 	,3.05 ],
                        'Er'	:	[	3.523 	,3.02 ],
                        'Tm'	:	[	3.019 	,3.01 ],
                        'Yb'	:	[	114.734 ,2.99 ],
                        'Lu'	:	[	20.632 	,3.24 ],
                        'Ta'	:	[	40.761 	,2.82 ],
                        'Re'	:	[	33.213 	,2.63 ],
                        'Os'	:	[	18.619 	,2.78 ],
                        'Ir'	:	[	36.735 	,2.53 ],
                        'Hg'	:	[	193.740 ,2.41 ],
                        'Tl'	:	[	342.190 ,3.87 ],
                        'Po'	:	[	163.547 ,4.20 ],
                        'At'	:	[	142.914 ,4.23 ],
                        'Rn'	:	[	124.799 ,4.25 ],
                        'Fr'	:	[	25.161 	,4.37 ],
                        'Ra'	:	[	203.301 ,3.28 ],
                        'Ac'	:	[	16.606 	,3.10 ],
                        'Th'	:	[	13.084 	,3.03 ],
                        'Pa'	:	[	11.071 	,3.05 ],
                        'Np'	:	[	9.561 	,3.05 ],
                        'Pu'	:	[	8.052 	,3.05 ],
                        'Am'	:	[	7.045 	,3.01 ],
                        'Cm'	:	[	6.542 	,2.96 ],
                        'Bk'	:	[	6.542 	,2.97 ],
                        'Cf'	:	[	6.542 	,2.95 ],
                        'Es'	:	[	6.039 	,2.94 ],
                        'Fm'	:	[	6.039 	,2.93 ],
                        'Md'	:	[	5.535 	,2.92 ],
                        'No'	:	[	5.535 	,2.89 ],
                        'Lw'	:	[	5.535 	,2.88 ],
                        'Ar'	:	[	119.800 ,3.34 ],
                        'Kr'	:	[	162.580 ,3.63 ],
                        'Xe'	:	[	226.140 ,3.95 ]}



eps_bin_means_manually_define = [2.52 	,
                                3.02 	,
                                3.52 	,
                                4.03 	,
                                4.53 	,
                                5.03 	,
                                5.54 	,
                                6.04 	,
                                6.54 	,
                                7.05 	,
                                7.55 	,
                                8.05 	,
                                8.55 	,
                                9.56 	,
                                10.90 	,
                                11.07 	,
                                12.58 	,
                                13.08 	,
                                15.10 	,
                                16.61 	,
                                17.61 	,
                                18.12 	,
                                18.62 	,
                                19.63 	,
                                20.13 	,
                                20.63 	,
                                21.14 	,
                                22.14 	,
                                22.65 	,
                                24.15 	,
                                25.16 	,
                                26.67 	,
                                28.18 	,
                                29.69 	,
                                33.21 	,
                                33.72 	,
                                34.72 	,
                                36.23 	,
                                36.48 	,
                                36.74 	,
                                37.43 	,
                                40.26 	,
                                40.76 	,
                                42.77 	,
                                47.86 	,
                                48.15 	,
                                55.86 	,
                                62.40 	,
                                114.73 	,
                                118.26 	,
                                119.77 	,
                                119.80 	,
                                124.80 	,
                                142.56 	,
                                142.91 	,
                                156.00 	,
                                161.03 	,
                                162.58 	,
                                163.55 	,
                                173.11 	,
                                183.17 	,
                                186.19 	,
                                193.74 	,
                                201.29 	,
                                203.30 	,
                                206.32 	,
                                216.38 	,
                                226.14 	,
                                256.64 	,
                                260.66 	,
                                276.77 	,
                                286.84 	,
                                333.63 	,
                                342.19 	]



def g_mean(x):
    np.seterr(divide = 'ignore') 
    if 0 not in x:
        mean = np.exp(np.log(x).mean())
    else:
        mean = 0 
    return mean


def norm(a1,a2,a3):
    return (a1**2 + a2**2 + a3**2)**0.5


def get_distance_perodic_cartesian(vec1, vec2, unit_cell):
    # calculating distances
    delta_a = vec1[0] - vec2[0]
    delta_b = vec1[1] - vec2[1]
    delta_c = vec1[2] - vec2[2]

    if delta_a > 0.5:
       vec2[0] = vec2[0] + 1.0
    elif delta_a < -0.5:
       vec2[0] = vec2[0] - 1.0

    if delta_b > 0.5:
       vec2[1] = vec2[1] + 1.0
    elif delta_b < -0.5:
       vec2[1] = vec2[1] - 1.0

    if delta_c > 0.5:
       vec2[2] = vec2[2] + 1.0
    elif delta_c < -0.5:
       vec2[2] = vec2[2] - 1.0
       
    FractionalDistVector = vec1 - vec2
    CartesianDistVector =  np.dot(np.transpose(unit_cell), FractionalDistVector)
    CaresianDistance = norm(CartesianDistVector[0],CartesianDistVector[1],CartesianDistVector[2])

    return  CaresianDistance


def get_distance_perodic_fractional(vec1, vec2):
    # calculating distances
    delta_a = vec1[0] - vec2[0]
    delta_b = vec1[1] - vec2[1]
    delta_c = vec1[2] - vec2[2]

    if delta_a > 0.5:
       vec2[0] = vec2[0] + 1.0
    elif delta_a < -0.5:
       vec2[0] = vec2[0] - 1.0

    if delta_b > 0.5:
       vec2[1] = vec2[1] + 1.0
    elif delta_b < -0.5:
       vec2[1] = vec2[1] - 1.0

    if delta_c > 0.5:
       vec2[2] = vec2[2] + 1.0
    elif delta_c < -0.5:
       vec2[2] = vec2[2] - 1.0
       
    FractionalDistVector = vec1 - vec2
    FractionalDistance = norm(FractionalDistVector[0],FractionalDistVector[1],FractionalDistVector[2])

    return  FractionalDistance


def grid_creation(a, b, c, grid_res, unit_vector):

    # Number of grid points 
    n_pts_in_a = int(a / float(grid_res)) + 1
    n_pts_in_b = int(b / float(grid_res)) + 1
    n_pts_in_c = int(c / float(grid_res)) + 1

    #Building the  Grid
    grid_coord_fractional = []
    grid_coord_Cartesian = []

    ngridpoints = 0
    for l in range(0,n_pts_in_a):
        for m in range(0,n_pts_in_b):
            for n in range(0,n_pts_in_c):
                ngridpoints = ngridpoints + 1
                grid_list = [l*float(grid_res)/a,  m*float(grid_res)/b, n*float(grid_res)/c]
                
                grid_coord_fractional.append(grid_list)
                grid_coord_Cartesian.append(np.dot(np.transpose(unit_vector), np.array(grid_list).T))
                
    grid_coord_fractional = np.reshape(grid_coord_fractional, (ngridpoints,3))
    grid_coord_Cartesian = np.reshape(grid_coord_Cartesian, (ngridpoints,3))
    

    return ngridpoints, grid_coord_fractional, grid_coord_Cartesian


def bin_means(data,slices,digitized_data, distribution_mean = True):
    
    bin_means = []
    for j in range(1, len(slices)):
        values = data[digitized_data == j]
        if len(values) == 0 :
            m = (slices[j-1] + slices[j])/2
        else :
            if distribution_mean:
                m = values.mean()
            else:
                m = (slices[j-1] + slices[j])/2
            
        bin_means.append(m)

    return bin_means


def Data_digitized(valueslist, step = 1, _max = None, _min=None):
    # avoid the input values in the forms other than np.array
    valueslist  = np.array(valueslist)
    # calculate and round max and min values
    _valuemax, _valuemin = max(valueslist), min(valueslist)
    # select the max and min values for slicing
    if (_max == None) and (_min == None):  
        print(f'Apply rounded(integer) max:{_valuemax} and min:{_valuemin} with step:{step} in slicing')
        _max, _min = math.ceil(_valuemax), math.floor(_valuemin)
    
    _slice = np.arange(_min, _max + step, step)
    _digitized_label = np.digitize(valueslist, _slice)
    _bin_means_binAvg = bin_means(valueslist,_slice,_digitized_label,distribution_mean = False)
    _bin_means_valueAvg = bin_means(valueslist,_slice,_digitized_label,distribution_mean = True)

    return _digitized_label, _bin_means_binAvg, _bin_means_valueAvg, _slice, _valuemax, _valuemin


def Data_digitized_eps(eps_list, eps_bin_mean):

    binResults = []
    for specific_value in eps_list:
        # Find the difference between the integer part of values and the specific value
        differences = np.abs(np.array(eps_bin_mean) - specific_value)
        # Get the value from your list that is closest to the specific value
        closest_value = eps_bin_mean[np.argmin(differences)]
        binResults.append(closest_value)
    print(f'eps max:{max(eps_list)}')
    binResults = np.array(binResults).reshape((len(binResults),1))
    return binResults
        
def Data_digitized_simple(valuelist, start, stop, num, string):
    bin_mean = np.linspace(start, stop, num)
    binResults = []
    for specific_value in valuelist:
        # Find the difference between the integer part of values and the specific value
        differences = np.abs(np.array(bin_mean) - specific_value)
        # Get the value from your list that is closest to the specific value
        closest_value = bin_mean[np.argmin(differences)]
        binResults.append(closest_value)
    print(f'{string} max:{max(valuelist)}')
    binResults = np.array(binResults).reshape((len(binResults),1))
    
    return binResults, bin_mean


def digitized_coords(digitized_label, bin_means):
    listdigitized = []
    length_means = len(bin_means)
    for i in digitized_label:
        '''The digitize is counting from 1 (first bin, the mean is counted from 0) to final block'''
        if i <= length_means:
            listdigitized.append(bin_means[i-1])
            
        else: #if the d is longer than 20 Å, append it to last bin with the mean 19.5
            listdigitized.append(bin_means[-1])
            
    listdigitized = np.array(listdigitized).reshape((len(listdigitized),1))
    return listdigitized

def BrutalSearch(grid_coord_f, atomcoords_f, unit_cell):
    BrutalDistance = []
    for coord in atomcoords_f:#cartesian
        distance = get_distance_perodic_cartesian(grid_coord_f, coord, unit_cell)
        BrutalDistance.append(distance)
    min_d  = min(BrutalDistance)
    min_ind = BrutalDistance.index(min_d)
    
    return min_d, min_ind

def min_d_ind_Brut(gridcood_f, atomindices, atomcoords_f, unit_vector):
    dists = []
    for ind in atomindices:
        distance = get_distance_perodic_cartesian(gridcood_f, atomcoords_f[ind], unit_vector)
        dists.append(distance)
    min_d = min(dists)
    min_ind = dists.index(min_d)
    
    return min_d, atomindices[min_ind]


def Bin_density(AllGridPoints, Data,
                Bin_means_d, Bin_means_Y, string = ''):
    
    print(f'Found grid points {len(Data)} | {AllGridPoints} from {string} data')
    data_ML = []
    # assign average epsilon and sigma values; based on all the points in each bin
    for j,k in itertools.product(Bin_means_d, Bin_means_Y):# iterate bin_means_binAvgs
        # extract the bin has the same d and same q values
        arr = Data[np.where((Data[:,0] == j) * (Data[:,1] == k))]
        Npoints = len(arr) # how many points in each bin
        if (Npoints != 0): # if the there are more than 1 point
            '''distance, charge or eps or sigma, density'''
            info_ML = [j,k,Npoints/AllGridPoints]
        else :
            info_ML = [j,k,0]  
            
        data_ML.append(info_ML)
        
    data_ML = np.array(data_ML)
    
    return data_ML



def alphaRescale(inputdata):
    alphalist = []
    for i in inputdata:
        if i != 0:
            alpha = 1
        else:
            alpha = 0
        alphalist.append(alpha)
        
    return np.array(alphalist)


def minorGrid(maingrid):
    minorgrid = []
    for c in range(len(maingrid) - 1):
        minor = (maingrid[c]+ maingrid[c+1])/2
        minorgrid.append(minor)
        
    return minorgrid


def _configure_distance_xaxis(ax, d_slice, distance_bin_step):
    """Set x limits and ticks from distance bin edges (d_slice) and bin width."""
    d_slice = np.asarray(d_slice, dtype=float)
    xmin = float(np.min(d_slice))
    xmax = float(np.max(d_slice))
    ax.set_xlim(xmin, xmax)
    span = xmax - xmin
    if span >= 18:
        major_step = 5.0
    elif span >= 10:
        major_step = 2.0
    else:
        major_step = max(float(distance_bin_step), 1.0)
    majors = np.arange(0.0, xmax + 1e-9 + major_step, major_step)
    majors = majors[(majors >= xmin - 1e-9) & (majors <= xmax + 1e-9)]
    if len(majors) == 0:
        majors = np.array([xmin, xmax])
    ax.set_xticks(majors)
    ax.xaxis.set_minor_locator(MultipleLocator(float(distance_bin_step)))
    ax.xaxis.grid(which="major", color="k", linewidth=0.12, alpha=0.55)
    ax.xaxis.grid(which="minor", color="k", linewidth=0.08, alpha=0.4)


def DetailDistributionPlotting(cifname,data_Q, data_eps, data_sigma, 
                               Bin_means_d,
                               Bin_means_q,
                               Bin_means_eps, 
                               Bin_means_sigma,
                               d_slice,
                               distance_bin_step,
                               xaxislength = 10, ylengthRatio = 1.1,
                               s_size = 70):
    
    
    '''horizonal subplots'''
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(xaxislength,xaxislength*ylengthRatio), dpi=300, layout="constrained")
    d_slice = np.asarray(d_slice, dtype=float)
    xmin = float(np.min(d_slice))
    xmax = float(np.max(d_slice))
    x_annot = xmin + 0.82 * (xmax - xmin)
    xlabel_d = rf"Distance [$\AA$] ($\Delta r$ = {float(distance_bin_step):g} $\AA$)"
    
    '''q plot'''
    _configure_distance_xaxis(ax1, d_slice, distance_bin_step)
    #process y direction
    ax1.set_ylim(-3,3)
    ax1.set_yticks(Bin_means_q[::2], minor=False)
    ax1.set_yticks(minorGrid(Bin_means_q), minor=True)
    ax1.yaxis.grid(which="minor", color='k', linewidth=0.1, alpha = 0.5)
    ax1.tick_params(axis='y', labelsize=8)
    # label and text
    ax1.set_xlabel(xlabel_d)
    ax1.text(x_annot, Bin_means_q[-5], s='Q',fontsize=30, weight = 'heavy')
    # scatter plot 
    data_ploting = data_Q
    Density = data_ploting[:,2]
    sc = ax1.scatter(data_ploting[:,0],data_ploting[:,1],c=Density,marker="s",s=s_size,alpha = alphaRescale(Density),cmap=cm.viridis)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    '''epsilon plot'''
    # fig, ax = plt.subplots(figsize=(4,4*len(q_slice)/len(d_slice)), dpi=300, layout="constrained")
    _configure_distance_xaxis(ax2, d_slice, distance_bin_step)
    # y direction
    yindex = [np.where(Bin_means_eps == val)[0][0] for val in data_eps[:,1]]
    yticks = np.arange(0,len(Bin_means_eps),1)
    ax2.set_ylim(0,max(yindex))
    ax2.set_yticks(yticks[::2], minor=False)
    ax2.set_yticklabels(Bin_means_eps[::2],fontsize=8)
    ax2.set_yticks(minorGrid(yticks), minor=True)
    ax2.yaxis.grid(which="minor", color='k', linewidth=0.1, alpha = 0.5)
    # label and text
    ax2.set_xlabel(xlabel_d)
    ax2.text(x_annot,yindex[-5], s='ε',fontsize=40, weight = 'heavy')
    # scatter plot 
    data_ploting = data_eps
    data_eps[:,1] = np.array(yindex)
    Density = data_ploting[:,2]
    sc = ax2.scatter(data_ploting[:,0],data_ploting[:,1],c=Density,marker="s",s=s_size,alpha = alphaRescale(Density),cmap=cm.viridis)
    
    '''sigma plot'''
    # fig, ax = plt.subplots(figsize=(4,4*len(q_slice)/len(d_slice)), dpi=300, layout="constrained")
    _configure_distance_xaxis(ax3, d_slice, distance_bin_step)
    # y direction
    ax3.set_ylim(min(Bin_means_sigma)*1.01,max(Bin_means_sigma))
    ax3.set_yticks(Bin_means_sigma[::2], minor=False)
    ax3.set_yticks(minorGrid(Bin_means_sigma), minor=True)
    ax3.yaxis.grid(which="minor", color='k', linewidth=0.1, alpha = 0.5)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.set_xlabel(xlabel_d)
    ax3.text(x_annot,Bin_means_sigma[-5], s='σ',fontsize=40,weight = 'heavy')
    # scatter plot 
    data_ploting = data_sigma
    Density = data_ploting[:,2]
    sc = ax3.scatter(data_ploting[:,0],data_ploting[:,1],c=Density,marker="s",s=s_size,alpha = alphaRescale(Density),cmap=cm.viridis)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    fig.savefig(f"./2DOutput/{cifname}/2Dhistogram.png",dpi=300)
    
    
    
    
    return


def KDtreeAnalyzer(cif = 'OPT_tty_sym_8_mc_9_sym_3_on_2_1B_2OH_1B_2OH.cif',  
                   cutoff = 'min',
                   grid_res = 1, RadiusRatio = 2.5, 
                   distance_bin_step = 1.0,
                   double_distance_binning = False,
                   BrutalChecking = False,
                   plot= True):

    '''Announce input parameters'''
    print(f'Cutoff : {cutoff}')
    print(f'Grid distance : {grid_res} Å')
    print(f'Searching ratio : {RadiusRatio}')
    print(f'Distance bin step : {distance_bin_step} Å')
    print(f'Double distance binning : {double_distance_binning}')
    print(f'Butal Checking : {BrutalChecking}')
    grid_res = float(grid_res)
    RadiusRatio = float(RadiusRatio)
    distance_bin_step = float(distance_bin_step)
    if distance_bin_step <= 0:
        raise ValueError(f'distance_bin_step must be > 0, got {distance_bin_step}')
    if double_distance_binning:
        distance_bin_step = distance_bin_step / 2.0
    ## Reading cif and 
    start_time1 = time.time()
    structure   = read(cif,store_tags=True)
    structure.set_pbc(np.array([ True,  True,  True])) # force pbc conditions
    ## Calling structure properties
    cell        = structure.cell.cellpar() 
    unitvector  = structure.get_cell().array
    SiteCharges = structure.info['_atom_site_charge'] 
    ChemicalSym = structure.get_chemical_symbols()
    CartesianCoordinates = structure.get_positions(wrap=False)
    FractionalCoordinates = structure.get_scaled_positions(wrap=True)
    '''Create grid points'''
    ngridpoints, grid_coord_fractional, grid_coord_cartesian = grid_creation(a = cell[0], b=cell[1], c=cell[2], grid_res =  grid_res, unit_vector=unitvector)
    print(f'Create {ngridpoints} grid points')
    print("---Reading cif and create grid points %s seconds ---" % (time.time() - start_time1))
    
    '''Create/Applied KD-trees'''
    start_time2 = time.time()
    'NAMD kdtree : Read cell lengths and without angles; the triclinic crystal can only be used with fractional coordinates'
    TreeP_NAMD = PeriodicCKDTree(bounds = np.array([1, 1, 1]), data = FractionalCoordinates, leafsize=1)
    'MDAnalysis kdtree : Read cartesian coordinates'
    _max = round(max(cell[0:3])/2, 2)
    _min = round(min(cell[0:3])/2, 2)
    
    if cutoff == 'max':
        cutoff = _max
    elif cutoff == 'min':
        cutoff = _min
        
    print(f'Applied cutoff {cutoff} Å in cartesian decision tree, max:{_max} | min:{_min}')
    TreeP_MDA = PeriodicKDTree(box = np.array(cell).astype('float32'), leafsize= 1)
    TreeP_MDA.set_coords(coords=CartesianCoordinates, cutoff=cutoff)
    
    'Counting'
    MDA = 0
    BS = 0
    BS_MDA = 0
    inconsistancelist = []
    neighbordict = {}
    
    for c, point in enumerate(grid_coord_fractional): 
    
        # NAMD nearset point for radius and cutoff defination(fractional coordinates)
        d_NAMD_fractional, ind_NAMD = TreeP_NAMD.query(x = point, k=1, eps=0, p=3)
        d_NAMD = get_distance_perodic_cartesian(point, FractionalCoordinates[ind_NAMD], unitvector)
        
        # MDA ball search:cartesian points  
        # '''TreeP_MDA.aug does not change with new curoff'''
        r_search = math.ceil(d_NAMD*RadiusRatio)
        # if r_search > cutoff:
        #     r_search = cutoff
        TreeP_MDA.cutoff = r_search # dynamic cutoff along the d_NAMD to avoid searching limitation 
        inds_MDA = TreeP_MDA.search(centers = grid_coord_cartesian[c], radius=r_search) # CartesianGpoint =  np.dot(np.transpose(unitvector), point)
        
        if len(inds_MDA) != 0:
            
            d_MDA, ind_MDA = min_d_ind_Brut(gridcood_f = point, atomindices = inds_MDA, atomcoords_f = FractionalCoordinates, unit_vector=unitvector)
            min_d_ind = ind_MDA
            min_d     = d_MDA
        else:
            d_Brutal, ind_Brutal = BrutalSearch(grid_coord_f = point,atomcoords_f = FractionalCoordinates, unit_cell=unitvector)
            min_d_ind = ind_Brutal
            min_d     = d_Brutal
        
        # Brutal search  
        if BrutalChecking:
            d_Brutal, ind_Brutal = BrutalSearch(grid_coord_f = point,atomcoords_f = FractionalCoordinates, unit_cell=unitvector)
            
            # Brutal checking 
            if ind_Brutal == ind_MDA:
                BS_MDA += 1
            else:
                if d_MDA > d_Brutal:
                    BS += 1
                else:
                    MDA += 1
                inconsistancelist.append([d_Brutal,d_MDA])
                
        
        # collect data into dictionary    
        min_d_q   = SiteCharges[min_d_ind]
        LJ_para   = LJParameters[ChemicalSym[min_d_ind]]
    
        neighbordict[c] = {}
        neighbordict[c]['index'] = min_d_ind
        neighbordict[c]['min_d'] = min_d
        neighbordict[c]['symbol'] = ChemicalSym[min_d_ind]
        neighbordict[c]['Charge'] = min_d_q
        neighbordict[c]['LJ-epsilon'] = LJ_para[0]
        neighbordict[c]['LJ-sigma'] = LJ_para[1]
        
    if BrutalChecking:
        inaccuracy = round(BS / ngridpoints * 100, 2)
        accuracy   = round(BS_MDA / ngridpoints * 100, 2)
        brutalfail = round(MDA / ngridpoints * 100, 2)
        print(f'BrutalChecking: {BrutalChecking} ({ngridpoints} points)')
        print(f'BS st MDA : {BS} | {ngridpoints} = {inaccuracy}%')
        print(f'BS bt MDA : {MDA} | {ngridpoints} = {brutalfail}%')
        print(f'BS eq MDA : {BS_MDA} | {ngridpoints} = {accuracy}% ')
        print('Inconsistant list')
        print(inconsistancelist)
    
    print("---Neighboring %s seconds ---" % (time.time() - start_time2))
    
    # Bin data (Distance and Charge)
    start_time3 = time.time()

    # Creast list for binning data
    list_min_symbo = [val.get('symbol') for val in neighbordict.values()]
    list_min_dist = [val.get('min_d') for val in neighbordict.values()]
    list_min_dist_q = [val.get('Charge') for val in neighbordict.values()]
    list_min_dist_eps = [val.get('LJ-epsilon') for val in neighbordict.values()]
    list_min_dist_sig = [val.get('LJ-sigma') for val in neighbordict.values()]

    # Bin distance data
    d_digitized_label, d_bin_means_binAvg, d_bin_means_valueAvg, d_slice, d_max, d_min = \
                Data_digitized(valueslist = list_min_dist, step = distance_bin_step, _max = 20, _min=0)
    digitized_coords_binAvg = digitized_coords(d_digitized_label, d_bin_means_binAvg)
    print(f'distance max:{max(list_min_dist)}')

    # Bin epsilon with assign mean of epsilon
    digitized_eps = Data_digitized_eps(eps_list = list_min_dist_eps, eps_bin_mean = eps_bin_means_manually_define)
    # Bin sigma data
    digitized_sigma, sigma_bin_mean = Data_digitized_simple(valuelist = list_min_dist_sig, 
                                                            start=2, stop=4.5, num = len(eps_bin_means_manually_define), 
                                                            string= 'sigma')
    # Bin charge data
    digitized_q, q_bin_mean  = Data_digitized_simple(valuelist = list_min_dist_q,
                                                     start=-3, stop=3, num = len(eps_bin_means_manually_define), 
                                                     string = 'charge')
    
    print(f'distance bin mean leangth : {len(d_bin_means_binAvg)}')
    print(f'charge bin mean leangth : {len(q_bin_mean)}')
    print(f'eps bin mean leangth : {len(eps_bin_means_manually_define)}')
    print(f'sigma bin mean leangth : {len(sigma_bin_mean)}')
    # Bin z direction density for q, eps and sigma
    d = digitized_coords_binAvg
    q = digitized_q 
    eps = digitized_eps
    sigma = digitized_sigma
    
    Data = np.column_stack((d,q))
    data_ML_d_q = Bin_density(AllGridPoints = ngridpoints, Data = np.column_stack((d,q)), 
                              Bin_means_d = d_bin_means_binAvg, Bin_means_Y = q_bin_mean, 
                              string='charge')
    

    data_ML_d_eps = Bin_density(AllGridPoints = ngridpoints, Data = np.column_stack((d,eps)), 
                              Bin_means_d = d_bin_means_binAvg, Bin_means_Y = eps_bin_means_manually_define, 
                              string='eps')

    
    data_ML_d_sigma = Bin_density(AllGridPoints = ngridpoints, Data = np.column_stack((d,sigma)), 
                              Bin_means_d = d_bin_means_binAvg, Bin_means_Y = sigma_bin_mean, 
                              string='sigma')
    
    Distance = data_ML_d_q[:,0]
    Density_Q = data_ML_d_q[:,2]
    Density_Eps = data_ML_d_eps[:,2]
    Density_Sig = data_ML_d_sigma[:,2]
    
    data_ML_density = np.column_stack((Density_Q, Density_Eps, Density_Sig))
    
    data_ML_all_xyz = np.column_stack((data_ML_d_q, data_ML_d_eps, data_ML_d_sigma))
                                     
                                       
    print("---Calculate Z direction (density) data %s seconds ---" % (time.time() - start_time3))
    
    #Plotting
    cifname = os.path.basename(cif)
    if plot:
        start_time4 = time.time()
        DetailDistributionPlotting(cifname=cifname, 
                                   data_Q = data_ML_d_q, 
                                   data_eps = data_ML_d_eps,
                                   data_sigma = data_ML_d_sigma,
                                   Bin_means_d= d_bin_means_binAvg,
                                   Bin_means_q = q_bin_mean,
                                   Bin_means_eps = eps_bin_means_manually_define,
                                   Bin_means_sigma = sigma_bin_mean,
                                   d_slice=d_slice,
                                   distance_bin_step=distance_bin_step,
                                 )
        
        print("---Plotting %s seconds ---" % (time.time() - start_time4))
    
    return data_ML_density, data_ML_all_xyz



def main(cif = 'OPT_tty_sym_8_mc_9_sym_3_on_2_1B_2OH_1B_2OH.cif', cutoffdef = 'min', 
         grid_distance = 1, SearchRadiusRatio = 2.5,
         distance_bin_step = 1.0, double_distance_binning = False,
         BrutalChecking=False, plotting = True, exportdata = False):
    
    print('========================Start========================')
    
    cifname = os.path.basename(cif)
    path = f"./2DOutput/{cifname}"
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    else :
        os.makedirs(path)
    print(f'Analyzing {cifname}')
    data_ML_density, data_ML_all_xyz = KDtreeAnalyzer(cif = cif,  
                       cutoff = cutoffdef,
                       grid_res = grid_distance, RadiusRatio = SearchRadiusRatio, 
                       distance_bin_step = distance_bin_step,
                       double_distance_binning = double_distance_binning,
                       BrutalChecking = BrutalChecking,
                       plot = plotting)
    
    
    
    if exportdata:
        
        np.save(file=f'./2DOutput/{cifname}/Avg_Density', arr = data_ML_density, allow_pickle=True, fix_imports=True)
        np.save(file=f'./2DOutput/{cifname}/Avg_all_xyz', arr =  data_ML_all_xyz, allow_pickle=True, fix_imports=True)
    print('========================End========================')
    
    # Create readable df 
    data_ML_density = pd.DataFrame(data_ML_density,columns=['Density_Q','Density_eps','Density_sig'])
    data_ML_avg_lj = pd.DataFrame(data_ML_all_xyz, columns=['Distance_bin_mean', 'Q_bin_mean', 'Density_Q', 'Distance_bin_mean','eps_bin_mean', 'Density_eps', 'Distance_bin_mean','sigma_bin_mean', 'Density_sigma'])
    
    return data_ML_density, data_ML_all_xyz
        
#################################################################
if __name__ == '__main__':
    
    data_ML_density, data_ML_all_xyz = main(cif = 'OPT_edq_sym_3_on_1_sym_7_mc_4_ntn_edge_1B_2SH.cif', cutoffdef = 'min', 
          grid_distance = 1, SearchRadiusRatio = 2.5,
          distance_bin_step = 1.0, double_distance_binning = False,
          BrutalChecking=False, plotting = True, exportdata = True) 

