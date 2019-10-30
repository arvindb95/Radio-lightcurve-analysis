import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

time0 = 2453218.7

fig = plt.figure(figsize=[10,4])

tab = Table.read("SN2004dk_data.txt",format="ascii")

times = tab["col1"].data 
fluxes = tab["col3"].data
err_fluxes = tab["col4"].data
freqs = tab["col2"].data

############### 4.9GHz Data #############
sel4 = np.where(freqs == 4.9)
time4 = times[sel4]
flux4 = fluxes[sel4]*10**(-3.0)
err_flux4 = err_fluxes[sel4]*10**(-3.0)

t = Table.read("lc4.9GHz_predicted.txt", format="ascii.csv")

str_data = t["FORPRINT: Tue Oct 29 14:08:26 2019"].data
pred_flux_4GHz = np.zeros(len(str_data))
pred_time_4GHz = np.zeros(len(str_data))

for i in range(len(str_data)):
    sep_data = str_data[i].split()
    pred_time_4GHz[i] = float(sep_data[0])
    pred_flux_4GHz[i] = float(sep_data[1])

ax1 = fig.add_subplot(121)

ax1.errorbar(time4, flux4, yerr=err_flux4, fmt="bo",markersize=3)
ax1.plot(pred_time_4GHz-time0, pred_flux_4GHz, "C1")
ax1.set_xscale("log")
ax1.set_yscale("log")
plt.ylim([0.1,10])
plt.xlim([8,1000])
ax1.set_xlabel(r"t-t$_{{0}}$ (d)")
ax1.set_ylabel("Flux (mJy)")
ax1.set_title("4.9GHz")

############### 8.5GHz Data #############
sel8 = np.where(freqs == 8.5)
time8 = times[sel8]
flux8 = fluxes[sel8]*10**(-3.0)
err_flux8 = err_fluxes[sel8]*10**(-3)

tab = Table.read("SN2004dk_data.txt",format="ascii")

flux_8GHz = tab["col3"].data[7:]*10**(-3)
flux_err_8GHz = tab["col4"].data[7:]*10**(-3)
time_8GHz = tab["col1"].data[7:]

t = Table.read("lc8.5GHz_predicted.txt", format="ascii.csv")

str_data = t["FORPRINT: Tue Oct 29 14:15:29 2019"].data
pred_flux_8GHz = np.zeros(len(str_data))
pred_time_8GHz = np.zeros(len(str_data))

for i in range(len(str_data)):
    sep_data = str_data[i].split()
    pred_time_8GHz[i] = float(sep_data[0])
    pred_flux_8GHz[i] = float(sep_data[1])

ax2 = fig.add_subplot(122)

ax2.errorbar(time8, flux8, yerr=err_flux8, fmt="bo", markersize=3)
ax2.plot(pred_time_8GHz-time0, pred_flux_8GHz, "C1")
ax2.set_xscale("log")
ax2.set_yscale("log")
plt.ylim([0.1,10])
plt.xlim([8,1000])
ax2.set_xlabel(r"t-t$_{{0}}$ (d)")
ax2.set_ylabel("Flux (mJy)")
ax2.set_title("8.5GHz")
plt.show()
