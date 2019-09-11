import numpy as np
import corner
import matplotlib.pyplot as plt
import emcee
from get_lc import *
import scipy.optimize as op
from astropy.table import Table
import time as ctime

t_start = ctime.time()

time = 10.0**np.arange(0,3.0,0.1)

th_obs_guess, ee_guess, eB_guess, pel_guess, nISM_guess = 35,0.06,0.0033,2.07,4.2e-3

### Published data

R3GHz_flux = np.array([0.0187, 0.0151, 0.0145, 0.0225, 0.0256, 0.034, 0.044, 0.0478, 0.061, 0.070])
R3GHz_flux_error = np.array([0.0063, 0.0039, 0.0037, 0.0034, 0.0029, 0.0036, 0.010, 0.006, 0.009, 0.0057])
R3GHz_time = np.array([16.42, 17.39, 18.33, 22.36, 24.26, 31.22, 46.26, 54.27, 57.22, 93.13])

scaling_factor = (3.0/8.0)**(-0.6) ## To scale 8GHz flux models to 3 GHz flux 

def lnlike(theta, time, flux, flux_error):
    th_obs, ee, eB, pel, nISM = theta
    model = scaling_factor*get_lc_R(time, th_obs, ee, eB, pel, nISM)
    inv_sigma2 = 1.0/(flux_error**2)
    return -(np.sum(((flux - model)**2)*inv_sigma2)) #- np.log(inv_sigma2)))

def lnprior(theta):
    th_obs, ee, eB, pel, nISM = theta
    if 0.5 < th_obs < 89.5 and 1.0e-2 < ee < 1.0e-1 and 1.0e-3 < eB < 1.0e-2 and 2.01 < pel < 2.19 and 1.0e-6 < nISM < 1.0e-1:
        return 0
    return -np.inf

def lnprob(theta, time, flux, flux_error):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, time, flux, flux_error)

method = 'L-BFGS-B'

nll = lambda *args: -lnlike(*args)
bnds = [(0.5,89.5),(0.01,0.1),(0.001,0.01),(2.01,2.19),(1e-6,1e-1)]
result = op.minimize(nll, [th_obs_guess, ee_guess, eB_guess, pel_guess, nISM_guess], bounds=bnds, args=(R3GHz_time, R3GHz_flux, R3GHz_flux_error), method=method)
print("The minimized parameters using "+method)
print(result['x'])

ml_t = Table(result['x'], names=["th_obs_ml", "ee_ml", "eB_ml", "pel_ml", "nISM_ml"])
ml_t.write("th_{t:0.0f}_ml_fit_params.txt".format(t=th_obs_guess),format="ascii", overwrite=True)

print("============================================================================")#

#### Enter emcee #####

ndim, nwalkers = 5, 500

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(R3GHz_time, R3GHz_flux, R3GHz_flux_error))

pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler.run_mcmc(pos, 500)

samples = sampler.chain[:, 200:, :].reshape((-1, ndim)) # taking all data after first 100 steps

th_obs_mcmc, ee_mcmc, eB_mcmc, pel_mcmc, nISM_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [0.2, 50, 99.8],axis=0)))    
print("The MCMC fit parameters and corresponding +1sigma and -1sigma values")
print(th_obs_mcmc, ee_mcmc, eB_mcmc, pel_mcmc, nISM_mcmc)
print("============================================================================")

t = Table([th_obs_mcmc, ee_mcmc, eB_mcmc, pel_mcmc, nISM_mcmc],names=["th_obs_mcmc", "ee_mcmc", "eB_mcmc", "pel_mcmc", "nISM_mcmc"])

t.write("th_{t:0.0f}_mcmc_fit_params.txt".format(t=th_obs_guess), format="ascii", overwrite=True)

## Plotting ###

fig = corner.corner(samples, bins=50, quantiles=(0.16,0.84), fill_contours = True, contourf_kwargs={'colors':['white','brown','orange','green']}, levels=[0, 1.0-np.exp(-0.5), 1.0-np.exp(-2.0), 1.0-np.exp(-4.5)], plot_datapoints=True, max_n_ticks=2 ,labels=[r"$\theta_{o}$",r"$\epsilon_{e}$",r"$\epsilon_{B}$",r"$p_{el}$",r"$n_{ISM}$"])

axes = np.array(fig.axes).reshape((ndim,ndim))

for i in np.array([1,2,4]):
    for j in range(ndim):
        if (j < i):
            ax = axes[i,j]
        #ax.set_xscale('log')
            ax.set_yscale('log')


for i in np.array([1,2,4]):
    for j in range(ndim):
        if (i < j):
            ax = axes[j,i]
        #ax.set_xscale('log')
            ax.set_xscale('log')

fig.savefig("th_{t:0.0f}_corner_plot.pdf".format(t=th_obs_guess))

guess_lc = get_lc_R(time,th_obs_guess, ee_guess, eB_guess, pel_guess, nISM_guess)
t_read = Table.read("th_{t:0.0f}_mcmc_fit_params.txt".format(t=th_obs_guess), format="ascii")
th_obs_mcmc, ee_mcmc, eB_mcmc, pel_mcmc, nISM_mcmc = t_read["th_obs_mcmc"].data[0], t_read["ee_mcmc"].data[0], t_read["eB_mcmc"].data[0], t_read["pel_mcmc"].data[0], t_read["nISM_mcmc"].data[0]
best_fit_lc = scaling_factor*get_lc_R(time, th_obs_mcmc, ee_mcmc, eB_mcmc, pel_mcmc, nISM_mcmc)

t_ml_read = Table.read("th_{t:0.0f}_ml_fit_params.txt".format(t=th_obs_guess), format="ascii")
#ml_lc = get_lc_R(time, t_ml_read["th_obs_ml"].data[0], t_ml_read["ee_ml"].data[0], t_ml_read["eB_ml"].data[0], t_ml_read["pel_ml"].data[0], t_ml_read["nISM_ml"].data[0])

fig3 = plt.figure()

no_of_curves = 500
sel_curves = samples[np.random.randint(len(samples), size=no_of_curves)]

th_obs = sel_curves[:,0]
ee = sel_curves[:,1]
eB = sel_curves[:,2]
pel = sel_curves[:,3]
nISM = sel_curves[:,4]
all_curves_tab = Table([th_obs, ee, eB, pel, nISM], names=["th_obs_arr", "ee_arr", "eB_arr", "pel_arr", "nISM_arr"])
all_curves_tab.write("th_{t:0.0f}_all_curves_table.txt".format(t=th_obs_guess), format="ascii", overwrite=True)

no_of_curves = len(th_obs)

curves_tab = Table.read("th_{t:0.0f}_all_curves_table.txt".format(t=th_obs_guess), format="ascii")
th_obs = curves_tab["th_obs_arr"].data 
ee = curves_tab["ee_arr"].data
eB = curves_tab["eB_arr"].data
pel = curves_tab["pel_arr"].data
nISM = curves_tab["nISM_arr"].data

############# Plotting confidence intervals ####################3

sig1_upper = np.zeros(len(time))
sig1_lower = np.zeros(len(time))
sig2_upper = np.zeros(len(time))
sig2_lower = np.zeros(len(time))
sig3_upper = np.zeros(len(time))
sig3_lower = np.zeros(len(time))

for ti in range(len(time)):
    t_flux = np.zeros(len(th_obs))
    for j in range(len(th_obs)):
        t_flux[j] = scaling_factor*get_lc_R(time[ti], th_obs[j], ee[j], eB[j], pel[j], nISM[j])
    conf_vals = np.percentile(t_flux,[0.2, 2.3, 15.9, 50, 84.1, 97.7, 99.8], axis=0) # 0.15, 2.5, 16, 50, 84, 97.5, 99.85
    sig3_lower[ti] = conf_vals[0]
    sig2_lower[ti] = conf_vals[1]
    sig1_lower[ti] = conf_vals[2]
    sig1_upper[ti] = conf_vals[4]
    sig2_upper[ti] = conf_vals[5]
    sig3_upper[ti] = conf_vals[6]

plt.fill_between(time, y1 = sig3_upper, y2 = sig3_lower,facecolor="brown", label=r"$3\sigma$")
plt.fill_between(time, y1 = sig2_upper, y2 = sig2_lower,facecolor="orange", label=r"$2\sigma$")
plt.fill_between(time, y1 = sig1_upper, y2 = sig1_lower,facecolor="green", label=r"$1\sigma$")
plt.loglog(time, best_fit_lc,"pink", label="mcmc")
plt.errorbar(R3GHz_time, R3GHz_flux, yerr=R3GHz_flux_error,fmt="C0o",markersize=4, elinewidth=0.7, label="Observations")
plt.title(r"Guess : $\theta_{{obs}}$ = {t:0.0f}$^{{o}}$, $\epsilon_{{e}}$ = {e:0.2E}, $\epsilon_{{B}}$ = {B:0.2E}, $p_{{el}}$ = {p:0.2f}, $n_{{ISM}}$ = {n:0.2E}".format(t=th_obs_guess, e = ee_guess, B = eB_guess, p = pel_guess, n = nISM_guess) + "\n" +
        r"Best fit : $\theta_{{obs}}$ = {t:0.0f}$^{{o}}$ , $\epsilon_{{e}}$ = {e:0.2E}, $\epsilon_{{B}}$ = {B:0.2E}, $p_{{el}}$ = {p:0.2f}, $n_{{ISM}}$ = {n:0.2E}".format(t=th_obs_mcmc, e = ee_mcmc, B = eB_mcmc, p = pel_mcmc, n = nISM_mcmc), fontsize=7, y=1.04)
plt.xlabel("Time since merger (Days)")
plt.ylabel("3GHz Flux (mJy)")
plt.legend()
plt.ylim(10**(-3), 1)
plt.xlim(1, 10**3)
plt.savefig("th{t:0.0f}_n_{n:0.0f}_m_".format(t=th_obs_guess, n=no_of_curves)+method+"_lc.pdf")

fig3, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.chain
labels = [r"$\theta_{{obs}}$", r"$\epsilon_{{e}}$", r"$\epsilon_{{B}}$", r"$p_{{el}}$", r"$n_{{ISM}}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

plt.savefig("test.pdf")

t_end = ctime.time()

t_final = t_end - t_start
print("Time taken for the code to run (s) : ",t_final)

