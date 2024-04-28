import numpy as np
import matplotlib.pyplot as plt
import pickle

# # Define the file names for your pickles
# pickle_files = ['results/loss1.pickle', 'loss2.pickle', 'loss3.pickle', 'loss4.pickle', 'loss5.pickle']

# Create empty lists to store the losses and standard errors
losses = []


# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/loss_nltgcr{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    losses.append(loss_data)
losses = np.array(losses)
mean_losses = np.mean(losses, axis=0)
standard_errors = np.std(losses, axis=0)/np.sqrt(5)


losses2 = []

# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/loss_nltgcr10{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    losses2.append(loss_data)

# Calculate the mean loss and standard error for each epoch
losses2 = np.array(losses2)
mean_losses10 = np.mean(losses2, axis=0)
standard_errors10 = np.std(losses2, axis=0)/np.sqrt(5)


losses2 = []

# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/loss_adah{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    losses2.append(loss_data)

# Calculate the mean loss and standard error for each epoch
losses2 = np.array(losses2)
mean_losses2 = np.mean(losses2, axis=0)
standard_errors2 = np.std(losses2, axis=0)/np.sqrt(5)

losses2 = []

# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/loss_adam{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    losses2.append(loss_data)

# Calculate the mean loss and standard error for each epoch
losses2 = np.array(losses2)
mean_losses3 = np.mean(losses2, axis=0)
standard_errors3 = np.std(losses2, axis=0)/np.sqrt(5)


losses2 = []

# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/loss_aw{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    losses2.append(loss_data)

# Calculate the mean loss and standard error for each epoch
losses2 = np.array(losses2)
mean_losses4 = np.mean(losses2, axis=0)
standard_errors4 = np.std(losses2, axis=0)/np.sqrt(5)

fig, ax = plt.subplots(figsize=(6, 4))
# Plot the mean loss vs epoch with standard error bars
plt.plot(range(len(mean_losses)), mean_losses, 'bo-', label='nlTGCR (m=1)', markevery=50)
plt.fill_between(range(len(mean_losses)), \
                 mean_losses - standard_errors, mean_losses + standard_errors, alpha=0.2)
plt.plot(range(len(mean_losses10)), mean_losses10, 'gv-', label='nlTGCR(m=10)', markevery=50)
plt.fill_between(range(len(mean_losses10)), \
                 mean_losses10 - standard_errors10, mean_losses10 + standard_errors10, alpha=0.2)
plt.plot(range(len(mean_losses2)), mean_losses2, '--', label='AdaHessian')
plt.fill_between(range(len(mean_losses2)), \
                 mean_losses2 - standard_errors2, \
                    mean_losses2 + standard_errors2, alpha=0.2)
plt.plot(range(len(mean_losses3)), mean_losses3, '-.', label='Adam')
plt.fill_between(range(len(mean_losses3)), \
                 mean_losses3 - standard_errors3, \
                    mean_losses3 + standard_errors3, alpha=0.2)
plt.plot(range(len(mean_losses4)), mean_losses4, ':', label='AdamW')
plt.fill_between(range(len(mean_losses4)), \
                 mean_losses4 - standard_errors4, \
                    mean_losses4 + standard_errors4, alpha=0.2)
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Training Loss',fontsize=14)
plt.legend(fontsize=14)
plt.savefig('trainloss.png',dpi=200,bbox_inches='tight')
plt.show()

#### Val Acc

accs = []


# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/acc_nltgcr{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    accs.append(loss_data)
accs = np.array(accs)
mean_acc = np.mean(accs, axis=0)
standard_errors = np.std(accs, axis=0)/np.sqrt(5)


accs = []
# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/acc_nltgcr10{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    accs.append(loss_data)
accs = np.array(accs)
mean_acc10 = np.mean(accs, axis=0)
standard_errors10 = np.std(accs, axis=0)/np.sqrt(5)

accs = []


# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/acc_adah{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    accs.append(loss_data)
accs = np.array(accs)
mean_acc2 = np.mean(accs, axis=0)
standard_errors2 = np.std(accs, axis=0)/np.sqrt(5)

accs = []


# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/acc_adam{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    
    # Append the losses to the list of losses
    accs.append(loss_data)
accs = np.array(accs)
mean_acc3 = np.mean(accs, axis=0)
standard_errors3 = np.std(accs, axis=0)/np.sqrt(5)

accs = []
# Loop through each pickle file
for i in range(5):
    # Load the loss data from the pickle
    file = f"results/acc_aw{i}.pkl"
    with open(file, 'rb') as f:
        loss_data = pickle.load(f)
    # Append the losses to the list of losses
    accs.append(loss_data)
accs = np.array(accs)
mean_acc4 = np.mean(accs, axis=0)
standard_errors4 = np.std(accs, axis=0)/np.sqrt(5)

fig, ax = plt.subplots(figsize=(6, 4))
# Plot the mean loss vs epoch with standard error bars
plt.plot(range(len(mean_acc)), mean_acc, 'bo-', label='nlTGCR (m=1)', markevery=40)
plt.fill_between(range(len(mean_acc)), \
                 mean_acc - standard_errors, mean_acc + standard_errors, alpha=0.2)
plt.plot(range(len(mean_acc10)), mean_acc10, 'gv-', label='nlTGCR(m=10)', markevery=40)
plt.fill_between(range(len(mean_acc10)), \
                 mean_acc10 - standard_errors10, mean_acc10 + standard_errors10, alpha=0.2)
plt.plot(range(len(mean_acc2)), mean_acc2, '--', label='AdaHessian')
plt.fill_between(range(len(mean_acc2)), \
                 mean_acc2 - standard_errors2, mean_acc2 + standard_errors2, alpha=0.2)
plt.plot(range(len(mean_acc3)), mean_acc3, '-.', label='Adam')
plt.fill_between(range(len(mean_acc3)), \
                 mean_acc3 - standard_errors3, mean_acc3 + standard_errors3, alpha=0.2)
plt.plot(range(len(mean_acc4)), mean_acc4, ':', label='AdamW')
plt.fill_between(range(len(mean_acc4)), \
                 mean_acc4 - standard_errors4, mean_acc4 + standard_errors4, alpha=0.2)
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Test Accuracy',fontsize=14)
plt.ylim([0.6,0.9])
plt.legend(fontsize=14)
plt.savefig('valacc.png',dpi=200,bbox_inches='tight')
plt.show()