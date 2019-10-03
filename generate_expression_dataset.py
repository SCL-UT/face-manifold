import os
import numpy as np
import scipy.io as sio
import random
from utils_for_3dmm import morphabel_model
from utils_for_3dmm.image_generation import img_from_params

sigma = 2 # SD of the Gaussian noise added to parameters
bfm = morphabel_model.MorphabelModel('utils_for_3dmm/BFM.mat')

print('Generating dataset...')
# dataset_generation
noise_parameter_indices = list(np.arange(29))
src_dir = 'dataset_generation/AFLW2000_AND_300W/'
params_size = len(noise_parameter_indices)
size = 50  # num of noisy versions according to each image of the dataset
noisy_ep_s = []
correct_ep_s = []
for file in os.listdir(src_dir):
    mat = sio.loadmat(src_dir+file)
    ep = mat['Exp_Para']
    correct_ep = list([ep]) * size
    noisy_ep = []
    for i in range(size):
        random_num = np.random.randint(params_size)
        selected_indices = random.sample(noise_parameter_indices, random_num+1)
        noisy_ep.append(bfm.add_noise_exp(ep, selected_indices, sigma=sigma))
    noisy_ep_s.append(noisy_ep)
    correct_ep_s.append(correct_ep)

# saving to file
noisy_ep_s = np.reshape(np.array(noisy_ep_s), (-1, 29))
correct_ep_s = np.reshape(np.array(correct_ep_s), (-1, 29))
train_noisy_ep_s = noisy_ep_s[:-30000, :]
train_correct_ep_s = correct_ep_s[:-30000, :]
test_noisy_ep_s = noisy_ep_s[-30000:, :]
test_correct_ep_s = correct_ep_s[-30000:, :]
save_folder_train = 'dataset_generation/dataset_expression/train/'
save_folder_test = 'dataset_generation/dataset_expression/test/'
if not os.path.exists(save_folder_train):
    os.makedirs(save_folder_train)
if not os.path.exists(save_folder_test):
    os.makedirs(save_folder_test)
np.save(save_folder_train + 'noisy.npy', train_noisy_ep_s)
np.save(save_folder_train + 'labels.npy', train_correct_ep_s)
np.save(save_folder_test + 'noisy.npy', test_noisy_ep_s)
np.save(save_folder_test + 'labels.npy', test_correct_ep_s)
print('Dataset generated succussfully.')

# dataset samples
print('Saving some sample images of the dataset...')
save_folder_samples = 'dataset_generation/dataset_expression/dataset_samples/'
if not os.path.exists(save_folder_samples):
    os.mkdir(save_folder_samples)
    
train_labels = np.load(save_folder_train + 'labels.npy')
train_noisy = np.load(save_folder_train + 'noisy.npy')
for i, file in enumerate(os.listdir(src_dir)):
    mat = sio.loadmat(src_dir+file)
    sp = mat['Shape_Para']
    tp = bfm.get_tex_para('zero')
    img_from_params(save_folder_samples, str(i) + '_label', bfm, sp,
                    train_labels[size*i].reshape(29, 1), tp)
    img_from_params(save_folder_samples, str(i) + '_noisy', bfm, sp,
                    train_noisy[size*i].reshape(29, 1), tp)
    if i == 4:
        break
print('Done.')