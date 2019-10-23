import numpy as np

def funF(x,p):
    y = 10.0**(-5.0+np.arange(1001)*(5.0+np.log10(20.0))/1.0e3)

    if (x > 20.0):
        x = y[1000]


