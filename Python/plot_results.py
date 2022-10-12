# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np


from matplotlib.ticker import MultipleLocator

fsize = 15
tsize = 18

tdir = 'in'

major = 5.0
minor = 3.0

style = 'default'

plt.style.use(style)
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor


loss_adam =[]
acc_adam = []
path = os.getcwd()
for i in range(1, 5):
    file1 = path +'\\GNN\\results\\loss_adam' + str(i) +'.pkl'
    with open(file1, "rb") as fp:  
        ll = pickle.load(fp)
    loss_adam.append(ll)
    file2 =  path +'\\GNN\\results\\acc_adam' + str(i) +'.pkl'
    with open(file2, "rb") as fp:  
        ll = pickle.load(fp)
    acc_adam.append(ll)
    
loss_adam_array = np.array(loss_adam)
loss_adam_mean = np.mean(loss_adam_array, axis=0)
loss_adam_std = np.std(loss_adam_array, axis=0)

acc_adam_array = np.array(acc_adam)
acc_adam_mean = np.mean(acc_adam_array, axis=0)
acc_adam_std = np.std(acc_adam_array, axis=0)


loss_adam1 =[]
acc_adam1 = []
path = os.getcwd()
for i in range(1, 5):
    file1 = path +'\\GNN\\results\\loss_adam1' + str(i) +'.pkl'
    with open(file1, "rb") as fp:  
        ll = pickle.load(fp)
    loss_adam1.append(ll)
    file2 =  path +'\\GNN\\results\\acc_adam1' + str(i) +'.pkl'
    with open(file2, "rb") as fp:  
        ll = pickle.load(fp)
    acc_adam1.append(ll)
    
loss_adam_array1 = np.array(loss_adam1)
loss_adam_mean1 = np.mean(loss_adam_array1, axis=0)
loss_adam_std1 = np.std(loss_adam_array1, axis=0)

acc_adam_array1 = np.array(acc_adam1)
acc_adam_mean1 = np.mean(acc_adam_array1, axis=0)
acc_adam_std1 = np.std(acc_adam_array1, axis=0)


loss_nltcgr =[]
acc_nltcgr = []
path = os.getcwd()
for i in range(1, 5):
    file1 = path +'\\GNN\\results\\loss_nltgcr' + str(i) +'.pkl'
    with open(file1, "rb") as fp:  
        ll = pickle.load(fp)
    loss_nltcgr.append(ll)
    file2 =  path +'\\GNN\\results\\acc_nltgcr' + str(i) +'.pkl'
    with open(file2, "rb") as fp:  
        ll = pickle.load(fp)
    acc_nltcgr.append(ll)
    
loss_nltcgr_array = np.array(loss_nltcgr)
loss_nltcgr_mean = np.mean(loss_nltcgr_array, axis=0)
loss_nltcgr_std = np.std(loss_nltcgr_array, axis=0)

acc_nltcgr_array = np.array(acc_nltcgr)
acc_nltcgr_mean = np.mean(acc_nltcgr_array, axis=0)
acc_nltcgr_std = np.std(acc_nltcgr_array, axis=0)

loss_nes =[]
acc_nes = []
path = os.getcwd()
for i in range(1, 5):
    file1 = path +'\\GNN\\results\\loss_nes' + str(i) +'.pkl'
    with open(file1, "rb") as fp:  
        ll = pickle.load(fp)
    loss_nes.append(ll)
    file2 =  path +'\\GNN\\results\\acc_nes' + str(i) +'.pkl'
    with open(file2, "rb") as fp:  
        ll = pickle.load(fp)
    acc_nes.append(ll)
    
loss_nes_array = np.array(loss_nes)
loss_nes_mean = np.mean(loss_nes_array, axis=0)
loss_nes_std = np.std(loss_nes_array, axis=0)

acc_nes_array = np.array(acc_nes)
acc_nes_mean = np.mean(acc_nes_array, axis=0)
acc_nes_std = np.std(acc_nes_array, axis=0)


fig, ax = plt.subplots(figsize=(7, 5))
iters = np.arange(0,len(loss_adam_mean))
# ax.plot(iters, loss_adam_mean,'g')
ax.errorbar(iters, loss_nes_mean, yerr = loss_nes_std, color='black',fmt ='-d',alpha=1,
            markevery=40,errorevery=10, label='Nesterov')
ax.errorbar(iters, loss_adam_mean, yerr = loss_adam_std, color='green',fmt ='-o',alpha=1,
            markevery=40,errorevery=10, label='Adam,lr=0.1')
ax.errorbar(iters, loss_adam_mean1, yerr = loss_adam_std1, color='green',fmt ='->',alpha=1,
            markevery=40,errorevery=10, label='Adam1,lr=0.01')
ax.errorbar(iters, loss_nltcgr_mean, yerr = acc_nltcgr_std, color='red',fmt ='-s',alpha=1,
            markevery=40,errorevery=10, label='NLTGCR')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.yscale('log')
plt.legend()
plt.savefig('gcn_loss.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')

fig, ax2 = plt.subplots(figsize=(7, 5))
iters = np.arange(0,len(loss_adam_mean))

ax2.errorbar(iters, acc_nes_mean, yerr = acc_nes_std, color='black',fmt ='-d',alpha=1,
            markevery=40,errorevery=10, label='Nesterov')
ax2.errorbar(iters, acc_adam_mean, yerr = acc_adam_std, color='green',fmt ='-o',alpha=1,
            markevery=40,errorevery=10, label='Adam,lr=0.1')
ax2.errorbar(iters, acc_adam_mean1, yerr = acc_adam_std1, color='green',fmt ='->',alpha=1,
            markevery=40,errorevery=10, label='Adam,lr=0.01')
ax2.errorbar(iters, acc_nltcgr_mean, yerr = acc_nltcgr_std, color='red',fmt ='-s',alpha=1,
            markevery=40,errorevery=10, label='NLTGCR')
plt.xlabel('Epoch')
plt.ylabel('Val Accuracy')
plt.ylim(0.5,1)
plt.legend()
plt.savefig('gcn_acc.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')


