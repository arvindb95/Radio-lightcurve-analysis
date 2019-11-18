import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

time_exp = 2455842.5

fig = plt.figure(figsize=[8,8])
plt.suptitle("Afterglow lightcurves for PTF11qcj (Palliyaguru et al. 2019)")

tab = Table.read("PTF11qcj_data_final_new.txt",format="ascii")

times = tab["col1"].data
fluxes = tab["col3"].data
err_fluxes = tab["col4"].data
freqs = tab["col2"].data

possible_freqs = np.array([2.5,3.5,5,7.4,13.5,16])

for n in range(len(possible_freqs)):
    sel = np.where(freqs == possible_freqs[n])
    time = times[sel]
    flux = fluxes[sel]*10**(-3.0)
    err_flux = err_fluxes[sel]*10**(-3.0)
    
    t = Table.read("lc"+str(possible_freqs[n])+"GHz_predicted.txt", format="ascii.csv")
    
    f = open("lc"+str(possible_freqs[n])+"GHz_predicted.txt","r")
    f1 = np.array(f.readlines())[0][:-1]
    
    str_data = t[f1].data
    pred_flux = np.zeros(len(str_data))
    pred_time = np.zeros(len(str_data))
    
    for i in range(len(str_data)):
        sep_data = str_data[i].split()
        pred_time[i] = float(sep_data[0])
        pred_flux[i] = float(sep_data[1])
    ax_plot_num = 321+n
    ax1 = fig.add_subplot(ax_plot_num)
    sel_times = np.where((pred_time-time_exp) <= 200) 
    pred_time = pred_time[sel_times]
    pred_flux = pred_flux[sel_times]
    ax1.errorbar(time, flux, yerr=err_flux, fmt="bo",markersize=2,capsize=2)
    ax1.plot(pred_time-time_exp, pred_flux, "C1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plt.ylim([0.1,10])
    plt.xlim([5,1000])
    plt.subplots_adjust(hspace = 0.5)
    if (n >= 4):
        ax1.set_xlabel(r"t-t$_{{0}}$ (d)")
    if (n%2 == 0): 
        ax1.set_ylabel("Flux (mJy)")
    ax1.set_title(str(possible_freqs[n])+"GHz",fontsize=8)

plt.savefig("PTF11qcj_fit.pdf")
