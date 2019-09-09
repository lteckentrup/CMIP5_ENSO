from netCDF4 import Dataset as open_ncfile
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

home ='/srv/ccrc/data02/z5227845/research/CMIP5_ENSO/data/'
folderOUT ='/srv/ccrc/data02/z5227845/research/CMIP5_ENSO/plots'

def plot(model, real, ax):
    
    ### open files
    atm = open_ncfile(home+'tas/historical/'+real+'/tas_'+model+ \
                      '_'+real+'_nino34.nc')
    ocean = open_ncfile(home+'tos/historical/'+real+'/tos_'+model+ \
                        '_'+real+'_nino34.nc')
        
    if model == 'HadGEM2':
        tos = ocean.variables['tos'][:1752,:,:]
        tas = atm.variables['tas'][:1752,:,:]
        
    else:
        tos = ocean.variables['tos'][:,:,:]
        tas = atm.variables['tas'][:,:,:]

    ### make arrays one dimensional for correlation  and scatterplot     
    tas = tas.flatten()
    tos = tos.flatten()

    ### make arrays two dimensional for linear regression  
    X = tas.reshape(-1, 1)  
    Y = tos.reshape(-1, 1)  
    
    ### Linear regression    
    reg = LinearRegression()  
    reg.fit(X, Y)  
    Y_pred = reg.predict(X)
    
    ### Information about linear regression model
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary())
    
    ### Pearson correlation
    corr, p_value = pearsonr(X, Y)
    
    ### Plot
    ax.scatter(tas, tos, marker='o', s=2, c='k')
    ax.plot(tas, Y_pred, label='Reg. coeff: '+str("%.2f" % reg.coef_[0][0])+
            '\nIntercept: '+ str("%.2f" % reg.intercept_[0]) + 
            '\nCorr. coeff: '+  str("%.2f" % corr[0]) + 
            '\nP-value: '+  str("%.2f" % p_value[0]))

    if ax in (ax3,ax4,ax5):
        ax.set_xlabel('Tas [K]')
    else:
        pass

    ax.set_xlim(287,304)
    ax.set_ylim(287,304)
    ax.set_ylabel('Tos [K]')
    ax.set_title(real)
    ax.legend(loc='best', frameon=False)
    plt.suptitle(model)

fig = plt.figure(1,figsize=(8.5,8.5))
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.85)
fig.subplots_adjust(top=0.91)   
fig.subplots_adjust(bottom=0.07)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.1)

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
    
gs = gridspec.GridSpec(3, 4)
ax1 = plt.subplot(gs[0,:2])
ax2 = plt.subplot(gs[0,2:])
ax3 = plt.subplot(gs[1,:2])
ax4 = plt.subplot(gs[1,2:])
ax5 = plt.subplot(gs[2,1:3])
    
real = ['r1i1p1','r2i1p1','r3i1p1','r4i1p1','r5i1p1']
axes = [ax1,ax2,ax3,ax4,ax5]

for r,a in zip(real,axes):  
    plot('HadGEM2', r, a)
    #plot('CanESM2', r, a)
    #plot('IPSL-CM5A', r, a)
    
plt.savefig('scatter_tas_tos.png', dpi=600)