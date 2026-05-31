"""
統計検定2023理工３を再現
"""
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = np.linspace(start = -1000, stop = 1000, num = 1000)
dx =x[1]-x[0]
p=0.3
mu1 = 100
mu2 = 200
sigma = 40

f1 = sp.stats.norm.pdf(x, loc = mu1, scale = sigma)
f2 = sp.stats.norm.pdf(x, loc = mu2, scale = sigma)

g = p * f1 + (1-p)*f2
G = np.zeros(len(x))

for i in range(len(x)):
    for j in range(i):
        G[i] += g[j] * dx

plt.plot(x, G)
plt.show()

SAMPLENUM=90

gen_s =np.random.rand(SAMPLENUM)
x_sample = np.zeros(SAMPLENUM)
idx = np.zeros(SAMPLENUM)

for i in range(SAMPLENUM):
    idx[i] = np.argmin(np.abs(np.array(gen_s[i]) - G))
    print(idx[i])
    x_sample[i] = x[int(idx[i])]




plt.plot(idx)
plt.show()

plt.plot(x_sample)
plt.show()

plt.hist(x_sample)
plt.show()

def gamma_hat(x, mu1, mu2, sigma, p):
    f1_ = sp.stats.norm.pdf(x, loc = mu1, scale = sigma)
    f2_ = sp.stats.norm.pdf(x, loc = mu2, scale = sigma)
    gamma_ = p * f1_ / ( p * f1_ + (1-p)*f2_ )
    return gamma_

# def p_hat(x_sample)
ITE =100
p_hat_array = np.zeros(ITE)
mu1_hat_array= np.zeros(ITE)
mu2_hat_array= np.zeros(ITE)
gamma_hat_array= np.zeros(ITE)

p_hat_pre = 0.5
mu1_hat_pre = 90
mu2_hat_pre = 150

for ite in range(ITE):
    p_hat = np.average( gamma_hat(x_sample, mu1_hat_pre, mu2_hat_pre, sigma, p_hat_pre) )
    mu1_hat = 1/p_hat_pre * np.average( gamma_hat(x_sample, mu1_hat_pre, mu2_hat_pre, sigma, p_hat_pre)*x_sample )
    mu2_hat = 1/(1-p_hat_pre) * np.average( (1-gamma_hat(x_sample, mu1_hat_pre, mu2_hat_pre, sigma, p_hat_pre)) * x_sample  )
    gamma_hat_array[ite] = np.prod(gamma_hat(x_sample, mu1_hat_pre, mu2_hat_pre, sigma, p_hat_pre))
    p_hat_array[ite]   = p_hat
    mu1_hat_array[ite] = mu1_hat
    mu2_hat_array[ite] = mu2_hat

    p_hat_pre = p_hat
    mu1_hat_pre = mu1_hat
    mu2_hat_pre = mu2_hat


plt.plot(p_hat_array)
plt.show()
    
plt.plot(mu1_hat_array)
plt.show()
plt.plot(mu2_hat_array)
plt.show()

plt.plot(gamma_hat_array)
plt.show()

