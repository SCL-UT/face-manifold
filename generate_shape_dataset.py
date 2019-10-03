import os
import numpy as np
import scipy.io as sio
import random
from utils_for_3dmm import morphabel_model
from utils_for_3dmm.image_generation import img_from_params

bfm = morphabel_model.MorphabelModel('utils_for_3dmm/BFM.mat')

print('Generating dataset...')
src_dir = 'dataset_generation/AFLW2000_AND_300W/'
noise_parameter_indices = list(np.arange(199))
size = 100  # num of noisy versions according to each image of the dataset
params_size = len(noise_parameter_indices)
noisy_sp_s = []
correct_sp_s = []
for file in os.listdir(src_dir):
    mat = sio.loadmat(src_dir+file)
    sp = mat['Shape_Para']
    correct_sp = list([sp]) * size
    noisy_sp = []
    for i in range(size):
        random_num = np.random.randint(params_size)
        selected_indices = random.sample(noise_parameter_indices, random_num+1)
        noisy_sp.append(bfm.add_noise_shape(sp, selected_indices, sigma=500000))
    noisy_sp_s.append(noisy_sp)
    correct_sp_s.append(correct_sp)

# saving to file
noisy_sp_s = np.reshape(np.array(noisy_sp_s), (-1, 199))
correct_sp_s = np.reshape(np.array(correct_sp_s), (-1, 199))
train_noisy_sp_s = noisy_sp_s[:-60000, :]
train_correct_sp_s = correct_sp_s[:-60000, :]
test_noisy_sp_s = noisy_sp_s[-60000:, :]
test_correct_sp_s = correct_sp_s[-60000:, :]
save_folder_train = 'dataset_generation/dataset_shape/train/'
save_folder_test = 'dataset_generation/dataset_shape/test/'
if not os.path.exists(save_folder_train):
    os.makedirs(save_folder_train)
if not os.path.exists(save_folder_test):
    os.makedirs(save_folder_test)
np.save('dataset_generation/dataset_shape/train/noisy.npy', train_noisy_sp_s)
np.save('dataset_generation/dataset_shape/train/labels.npy', train_correct_sp_s)
np.save('dataset_generation/dataset_shape/test/noisy.npy', test_noisy_sp_s)
np.save('dataset_generation/dataset_shape/test/labels.npy', test_correct_sp_s)

print('Dataset generated succussfully.')

# dataset samples
print('Saving some sample images of the dataset...')
save_folder_samples = 'dataset_generation/dataset_shape/dataset_samples/'
if not os.path.exists(save_folder_samples):
    os.mkdir(save_folder_samples)
train_labels = np.load(save_folder_train + 'labels.npy')
train_noisy = np.load(save_folder_train + 'noisy.npy')
for i, file in enumerate(os.listdir(src_dir)):
    mat = sio.loadmat(src_dir+file)
    ep = mat['Exp_Para']
    sp_l = train_labels[size*i]
    sp_n = train_noisy[size*i]
    tp = bfm.get_tex_para('zero')
    img_from_params(save_folder_samples, str(i) + '_label', bfm,
                    sp_l.reshape(199, 1), ep, tp)
    img_from_params(save_folder_samples, str(i) + '_noisy', bfm,
                    sp_n.reshape(199, 1), ep, tp)
    if i == 4:
        break
print('Done.')
