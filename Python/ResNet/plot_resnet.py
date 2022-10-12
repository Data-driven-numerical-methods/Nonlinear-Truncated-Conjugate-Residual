# -*- coding: utf-8 -*-

import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
from matplotlib import container

loss_adam = []
loss_nes = []  
loss_nltgcr = []
acc_nes = []
acc_nltgcr = []
acc_adam = []

for i in range(4):
    file1 = "results/acc_nes"+str(i+1)+".pkl"
    file2 = "results/loss_nes"+str(i+1)+".pkl"
    file3 = "results/acc_T"+str(i+1)+".pkl"
    file4 = "results/loss_T"+str(i+1)+".pkl"
    file5 = "results/acc_adam"+str(i+1)+".pkl"
    file6 = "results/loss_adam"+str(i+1)+".pkl"
    with open(file1, "rb") as fp:
        hist = pickle.load(fp)
        acc_nes.append(hist)
    with open(file2, "rb") as fp:
        hist = pickle.load(fp)
        loss_nes.append(hist)
    with open(file3, "rb") as fp:
        hist = pickle.load(fp)
        acc_nltgcr.append(hist)    
    with open(file4, "rb") as fp:
        hist = pickle.load(fp)
        loss_nltgcr.append(hist)
        
    with open(file5, "rb") as fp:
        hist = pickle.load(fp)
        acc_adam.append(hist)    
    with open(file6, "rb") as fp:
        hist = pickle.load(fp)
        loss_adam.append(hist)


loss_nes = np.array(loss_nes)
loss_nltgcr = np.array(loss_nltgcr)
loss_adam = np.array(loss_adam)
avg_nes = np.mean(loss_nes,axis=0)
avg_nltgcr = np.mean(loss_nltgcr,axis=0)
avg_adam = np.mean(loss_adam, axis=0)
std_nes = np.std(loss_nes,axis=0)
std_nltgcr = np.std(loss_nltgcr,axis=0)
std_adam = np.std(loss_adam,axis=0)

acc_nes = np.array(acc_nes)
acc_nltgcr = np.array(acc_nltgcr)
acc_adam = np.array(acc_adam)
avg_acc_nes = np.mean(acc_nes,axis=0)
avg_acc_nltgcr = np.mean(acc_nltgcr,axis=0)
std_acc_nes = np.std(acc_nes,axis=0)
std_acc_nltgcr = np.std(acc_nltgcr,axis=0)
avg_acc_adam = np.mean(acc_adam,axis=0)
std_acc_adam = np.std(acc_adam,axis=0)


fig, ax = plt.subplots(figsize=(7,5))

markevery = 10
x = np.arange(1, 51, 1)
ax.errorbar(x, avg_nes, yerr=std_nes, fmt='-k*',markevery=markevery,label='Nesterov')
ax.errorbar(x, avg_adam, yerr=std_adam,fmt='-go',markevery=markevery,label='Adam')
ax.errorbar(x, avg_nltgcr, yerr=std_nltgcr, fmt='-rs',markevery=markevery,label='NLTGCR')
ax.legend(fontsize=14)
plt.yscale('log')
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.set_xlabel('Epoch',fontsize=14)
ax.set_ylabel('Training Mean Squared Error',fontsize=14)
fig.savefig('loss_resnet.png', dpi=300)



fig, ax = plt.subplots(figsize=(7,5))
markevery = 10
x = np.arange(1, 51, 1)
ax.errorbar(x, avg_acc_nes, yerr=std_acc_nes, fmt='-k*',markevery=markevery,label='Nesterov')
ax.errorbar(x, avg_acc_adam, yerr=std_acc_adam,fmt='-go',markevery=markevery,label='Adam')
ax.errorbar(x, avg_acc_nltgcr, yerr=std_acc_nltgcr, fmt='-rs',markevery=markevery,label='NLTGCR')
ax.legend(fontsize=14)
# plt.yscale('log')
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.set_xlabel('Epoch',fontsize=14)
ax.set_ylabel('Test Top-1 Accuracy',fontsize=14)
fig.savefig('acc_resnet.png', dpi=300)