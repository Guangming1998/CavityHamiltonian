import sys
import numpy as np
import pandas as pd
import os # try to use 'os' module to create new file directory for output files.
from Cavity_ssh import Trajectory_SSHmodel
path = sys.argv[-1].replace('cavity.py','')
'check my path'
# print(path)

# directory_names = ['csv_output','dat_output']

# try:
#     for directory_name in directory_names:
#         try:
#             os.makedirs(directory_name)
#         except FileExistsError:
#             print(f"Error: '{directory_name}' already exists")

# except :
#     print("Error occure while creating the directories.")
plotResult = False
printOutput = False
SanityCheck = True

if '--print' in sys.argv:
    printOutput = True
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')

'unit conversion'
ps_to_au = 4.13414e4 # ps to atomic time
wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
Adot_to_au = 1.88973 #angstrom to atomic length
wavenumber_to_amuAps = 2180.66
hbar_to_amuAps = 1.1577e4
amu_to_auMass = 1822.88 #amu is Dalton mass!
kBT_to_au = 3.16681e-6 # has made mistake before!

if 'param.in' in path:
    exec(open('param.in').read())
    # atomic_unit = True
    
    # if atomic_unit:
    #     staticCoup = 300 *wavenumber_to_au
    #     dynamicCoup = 995/Adot_to_au * wavenumber_to_au
    #     kBT = 104.3*wavenumber_to_au
    #     mass = 100
    #     Kconst = 14500/(ps_to_au**2)
    #     hbar = 1
    #     dt = 0.025e-3*ps_to_au
    # else: #"use amu*A^2*ps^-2 unit"
    #     staticCoup = 300 *wavenumber_to_amuAps
    #     dynamicCoup = 995 * wavenumber_to_amuAps
    #     kBT = 104.3*wavenumber_to_amuAps
    #     mass = 100
    #     Kconst = 14500
    #     hbar = 1*hbar_to_amuAps
    #     dt = 0.025e-3
else:
    Ehrenfest = False # define a bool variable for Ehrenfest force switch
    atomic_unit = True #define a bool variable for atomic unit switch
    Runge_Kutta = True # define a bool variable for switch from velocity verlet to Runge Kutta.

    dt = 0.025e-4
    Ntimes =int(2/dt)
    Nskip = 1
    
    # hbar_to_amuAps = 1.1577e4
    
    #conversion 
    Nmol = 64
    
    if atomic_unit:
        staticCoup = 300 *wavenumber_to_au
        dynamicCoup = 995/Adot_to_au * wavenumber_to_au
        kBT = 300* kBT_to_au
        mass = 250*amu_to_auMass
        Kconst = 14500*amu_to_auMass/(ps_to_au**2)
        hbar = 1
        dt = dt*ps_to_au
        couplingstrength = 100 *wavenumber_to_au
        cavitydecayrate = 100*wavenumber_to_au
        cavityFrequency = -2*staticCoup - 1j*cavitydecayrate/2
    else: #"use amu*A^2*ps^-2 unit"
        staticCoup = 300 *wavenumber_to_amuAps
        dynamicCoup = 995 * wavenumber_to_amuAps
        kBT = 104.3*wavenumber_to_amuAps
        mass = 100
        Kconst = 14500
        hbar = 1*hbar_to_amuAps
        dt = 0.025e-3
    
    useDiagonalDisorder = False
    useCavityHamiltonian = True

model1 = Trajectory_SSHmodel(Nmol,hbar)
'check if param.in valids'
# print(model1.Nmol)
model1.initialGaussian(kBT,mass,Kconst)
model1.initialHamiltonian(staticCoup,dynamicCoup)

if useDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(1e-4*staticCoup)
    
if useCavityHamiltonian:
    model1.initialHamiltonianCavity(couplingstrength,cavityFrequency)

if useDiagonalDisorder or SanityCheck or useCavityHamiltonian:
    '''
    Test different case with different disorder
    '''
    # model1.initialCj_disorder()
    # model1.initialCj_disorder_trajectory(kBT)
    # model1.initialCj_cavity()
    model1.initialCj_cavity_trajectory(kBT)

# else:
    
    CJJavg1_list,CJJavg_cav_list, CJJavg_mol_list = [model1.getCurrentCorrelation_cavity()[2]], [model1.getCurrentCorrelation_cavity()[1]], [model1.getCurrentCorrelation_cavity()[0]]
    CJJavg1_list.pop()
    CJJavg_cav_list.pop()
    CJJavg_mol_list.pop()


    for i in range(Ntimes):
        model1.updateHmol()
        model1.updateHamiltonianCavity()
        model1.old_Aj(Ehrenfest=False)
        
        model1.RK4_Cj_cavity_trajectory(dt)
        model1.propagateJcav0Cj_RK4(dt)
        model1.propagateJtot0Cj_RK4(dt)
        model1.propagateJmol0Cj_RK4(dt)
        
        model1.velocityVerlet(dt,Ehrenfest=False)

            
        
        if i%Nskip ==0:
            element_mol, element_cav, element_total = model1.getCurrentCorrelation_cavity()
            
            CJJavg1_list.append(element_total)
            CJJavg_cav_list.append(element_cav)
            CJJavg_mol_list.append(element_mol)
    CJJavg1_list,CJJavg_cav_list,CJJavg_mol_list = np.array(CJJavg1_list,complex), np.array(CJJavg_cav_list,complex), np.array(CJJavg_mol_list,complex)
    
    dici = {'CJJ':CJJavg1_list,'CJJ_cav':CJJavg_cav_list,'CJJ_mol':CJJavg_mol_list}
    data = pd.DataFrame(dici)
    data.to_csv('CJJ_cavity.csv')
    





