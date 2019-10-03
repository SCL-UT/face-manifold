import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from training.network import AutoEncoder4
import os

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        noisy_param, label_param = sample['noisy_param'], sample['label_param']
        return {'noisy_param': torch.from_numpy(noisy_param).float(),
                'label_param': torch.from_numpy(label_param).float()}


class Normalize(object):   # data normalization between -1 and 1 and then converting to Tensor
    def __call__(self, sample):
        noisy_param, label_param = sample['noisy_param'], sample['label_param']
        noisy_param = noisy_param/1e5
        label_param = label_param/1e5
        return {'noisy_param': torch.from_numpy(noisy_param).float(),
                'label_param': torch.from_numpy(label_param).float()}


class Noisy3dmmDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.noisy_params = np.load(path + 'noisy.npy')
        self.label_params = np.load(path + 'labels.npy')
        self.transform = transform

    def __len__(self):
        return len(self.noisy_params)

    def __getitem__(self, idx):
        sample = {'noisy_param': self.noisy_params[idx, :], 'label_param': self.label_params[idx, :]}
        if self.transform:
            sample = self.transform(sample)
        return sample['noisy_param'], sample['label_param']

# Hyper Parameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001
data_path = 'dataset_generation/dataset_shape/'
input_length = 199
dataset = Noisy3dmmDataset(data_path + 'train/', transform=Normalize())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
net = AutoEncoder4()
criterion = nn.MSELoss() #nn.L1Loss()
opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

def model_error(noisy_params, label_params, net):
  error = 0
  net.cpu()
  for i, param in enumerate(noisy_params):
      param = torch.Tensor(param)
      param = param.view(-1, 1, input_length)
      out = net(param)
      out = out.detach().numpy()
      out = out.reshape(-1)
      l = label_params[i].reshape(-1)
      error += ((out - l)**2).mean()
  return error/len(label_params)


loss_train = []
error_val = []
test_noisy_params = np.load(data_path + 'test/noisy.npy')
test_label_params = np.load(data_path + 'test/labels.npy')
val_noisy = test_noisy_params[:1000]
val_label = test_label_params[:1000]
train_noisy = np.load(data_path + 'train/noisy.npy')
train_labels = np.load(data_path + 'train/labels.npy')
mins = np.amin(train_labels, axis=0)
maxs = np.amax(train_labels, axis=0)
test_noisy_params = test_noisy_params/1e5
test_label_params = test_label_params/1e5
val_noisy = val_noisy/1e5
val_label = val_label/1e5
save_folder = './trained_models/shape/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# training the  network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_train = []
for epoch in range(num_epochs):
    valid_error = model_error(val_noisy, val_label, net)
    if torch.cuda.is_available():
        net.cuda()
    error_val.append(valid_error)
    print('validation error: ', valid_error)
    del valid_error
    loss_batch = []
    for i, (noisy, labels) in enumerate(dataloader):
        noisy, labels = noisy.to(device), labels.to(device)
        # forward step
        opt.zero_grad()
        noisy = noisy.view(-1, 1, input_length)
        outputs = net(noisy)
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        # backward step
        loss.backward()
        # optimization step
        opt.step()
        if (i) % 1000 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item()))
        loss_batch.append(loss.item())
        del loss
        del outputs
        del labels
        del noisy
    loss_train.append(np.mean(loss_batch))
    del loss_batch
    e = epoch + 1
    if epoch == 0:
        torch.save(net.state_dict(),
                   save_folder + 'epoch_' + str(e) + '.pkl')
    else:
        if loss_train[-1] < loss_train[-2]:
            torch.save(net.state_dict(), save_folder + 'epoch_' + str(e) + '.pkl')

# plot loss
plt.figure(1)
plt.plot(np.arange(len(loss_train)) + 1, loss_train)
plt.title(' Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(save_folder + 'loss.png')
plt.show()


# plot accuracy
plt.figure(2)
plt.plot(np.arange(len(error_val)) + 1, error_val)
plt.title(' Error')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.savefig(save_folder + 'val_error.png')
plt.show()


np.save(save_folder + 'loss.npy', loss_train)
np.save(save_folder + 'validation_error.npy', error_val)

# test numerical
print('Test Error is: ', model_error(test_noisy_params, test_label_params, net))
