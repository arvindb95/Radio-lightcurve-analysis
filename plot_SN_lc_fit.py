import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

time0 = 2453218.7

fig = plt.figure(figsize=[8,8])
plt.suptitle("Afterglow lightcurves for SN2004dk (Wellons et al. 2012)")

tab = Table.read("SN2004dk_data_100days.txt",format="ascii")

times = tab["col1"].data 
fluxes = tab["col3"].data
err_fluxes = tab["col4"].data
freqs = tab["col2"].data

possible_freqs = np.array([4.9,8.5,15,22.5])

for n in range(len(possible_freqs)):
    sel4 = np.where(freqs == possible_freqs[n])
    time4 = times[sel4]
    flux4 = fluxes[sel4]*10**(-3.0)
    err_flux4 = err_fluxes[sel4]*10**(-3.0)
    
    t = Table.read("lc"+str(possible_freqs[n])+"GHz_predicted.txt", format="ascii.csv")
    
    f = open("lc"+str(possible_freqs[n])+"GHz_predicted.txt","r")
    f1 = np.array(f.readlines())[0][:-1]
    
    str_data = t[f1].data
    pred_flux_4GHz = np.zeros(len(str_data))
    pred_time_4GHz = np.zeros(len(str_data))
    
    for i in range(len(str_data)):
        sep_data = str_data[i].split()
        pred_time_4GHz[i] = float(sep_data[0])
        pred_flux_4GHz[i] = float(sep_data[1])
    ax_plot_num = 221+n
    ax1 = fig.add_subplot(ax_plot_num)
    
    ax1.errorbar(time4, flux4, yerr=err_flux4, fmt="bo",markersize=4,capsize=3)
    ax1.plot(pred_time_4GHz-time0, pred_flux_4GHz, "C1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plt.ylim([0.1,5])
    plt.xlim([5,1000])
    #ax1.set_xlabel(r"t-t$_{{0}}$ (d)")
    ax1.set_ylabel("Flux (mJy)")
    ax1.set_title(str(possible_freqs[n])+"GHz")

plt.savefig("SN2004dk_fit.pdf")
