import scipy.io as sio
from utils_for_3dmm import morphabel_model
from utils_for_3dmm.image_generation import img_from_params
import random
import torch
import numpy as np
import os
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt

bfm = morphabel_model.MorphabelModel('utils_for_3dmm/BFM.mat')


def denoise_shape_and_exp(vec, net_shape=None, net_exp=None, mode='both'):
    # This function takes 'vec' which is a vector of 3DMM parameters containing shape, expression or both duo to the
    # given 'mode'. The parameters are fed to the networks and the output denoised parameters are returned.
    if mode == 'both':
        sp = vec[:199]
        ep = vec[199:]

        sp = torch.from_numpy(sp).float()
        sp = sp.view(-1, 1, 199)
        sp_out = net_shape(sp)
        sp_out = sp_out.detach().numpy()
        sp_out = sp_out.reshape(199, 1)

        ep = torch.from_numpy(ep).float()
        ep = ep.view(-1, 1, 29)
        ep_out = net_exp(ep)
        ep_out = ep_out.detach().numpy()
        ep_out = ep_out.reshape(29, 1)

        out = np.concatenate((sp_out, ep_out))
        return out
    elif mode == 'shape':
        sp = vec
        sp = torch.from_numpy(sp).float()
        sp = sp.view(-1, 1, 199)
        sp_out = net_shape(sp)
        sp_out = sp_out.detach().numpy()
        sp_out = sp_out.reshape(199, 1)
        return sp_out
    elif mode == 'exp':
        ep = vec
        ep = torch.from_numpy(ep).float()
        ep = ep.view(-1, 1, 29)
        ep_out = net_exp(ep)
        ep_out = ep_out.detach().numpy()
        ep_out = ep_out.reshape(29, 1)
        return ep_out


def evaluate_test_data(net, size=20, src_dir='dataset_generation/AFLW2000_AND_300W/', sigma_shape=100000, sigma_exp=1,
                       mode='shape', save_folder='model_evaluation/output_on_test_data/'):
    # This function adds noise to 3dmm samples of a database exisiting in 'src_dir'. The noisy samples are fed to the
    # networks and denoised. The results are saved in 'save_folder'. 'mode' determines that the noise is added to
    # only the expression parameters, or only the shape parameters, or both of them.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    noise_parameter_indices_exp = list(np.arange(29))
    params_size_exp = len(noise_parameter_indices_exp)
    noise_parameter_indices_shape = list(np.arange(199))
    params_size_shape = len(noise_parameter_indices_shape)
    inputs = []
    outputs = []
    if mode == 'shape':
        if not os.path.exists(save_folder + 'shape/'):
            os.makedirs(save_folder + 'shape/')
        for i, file in enumerate(os.listdir(src_dir)):
            if i > 5736:  # database contains 5837 images. loading test images.
                mat = sio.loadmat(src_dir + file)
                sp = mat['Shape_Para']
                tp = bfm.get_tex_para('zero')
                ep = mat['Exp_Para']
                random_num = np.random.randint(params_size_shape)
                selected_indices = random.sample(noise_parameter_indices_shape, random_num + 1)
                sp_noisy = bfm.add_noise_shape(sp, selected_indices, sigma=sigma_shape)
                img_from_params(save_folder + 'shape', str(i+1) + '_gt', bfm, sp.reshape(199, 1), ep, tp)
                img_from_params(save_folder + 'shape', str(i+1) + '_input', bfm, sp_noisy.reshape(199, 1), ep, tp)
                out_shape = denoise_shape_and_exp(sp_noisy, net_shape=net, mode='shape')
                img_from_params(save_folder + 'shape', str(i+1) + '_output', bfm, out_shape, ep, tp)
                inputs.append((sp_noisy.reshape(199, 1), ep, tp))
                outputs.append((out_shape, ep, tp))
            if i == 5736 + size:
                break
        df = pd.DataFrame(inputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'shape/' + 'inputs.pkl')
        df = pd.DataFrame(outputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'shape/' + 'outputs.pkl')
    elif mode == 'exp':
        if not os.path.exists(save_folder + 'expression/'):
            os.makedirs(save_folder + 'expression/')
        for i, file in enumerate(os.listdir(src_dir)):
            if i > 5736:
                mat = sio.loadmat(src_dir + file)
                sp = mat['Shape_Para']
                tp = bfm.get_tex_para('zero')
                ep = mat['Exp_Para']
                random_num = np.random.randint(params_size_exp)
                selected_indices = random.sample(noise_parameter_indices_exp, random_num + 1)
                ep_noisy = bfm.add_noise_exp(ep, selected_indices, sigma=sigma_exp)
                img_from_params(save_folder + 'expression', str(i+1) + '_gt', bfm, sp,
                                ep.reshape(29, 1), tp)
                img_from_params(save_folder + 'expression', str(i+1) + '_input', bfm, sp,
                                ep_noisy.reshape(29, 1), tp)
                out = denoise_shape_and_exp(ep_noisy, net_exp=net, mode='exp')
                img_from_params(save_folder + 'expression', str(i+1) + '_output', bfm, sp,
                                out.reshape(29, 1), tp)
                inputs.append((sp, ep_noisy.reshape(29, 1), tp))
                outputs.append((sp, out.reshape(29, 1), tp))
            if i == 5736 + size:
                break
        df = pd.DataFrame(inputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'expression/' + 'inputs.pkl')
        df = pd.DataFrame(outputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'expression/' + 'outputs.pkl')
    elif mode == 'both':
        if not os.path.exists(save_folder + 'shape_and_expression/'):
            os.makedirs(save_folder + 'shape_and_expression/')
        net_shape = net[0]
        net_exp = net[1]
        sigma_shape = sigma_shape
        sigma_exp = sigma_exp
        for i, file in enumerate(os.listdir(src_dir)):
            if i > 5736:
                mat = sio.loadmat(src_dir + file)
                sp = mat['Shape_Para']
                tp = bfm.get_tex_para('zero')
                ep = mat['Exp_Para']
                random_num = np.random.randint(params_size_shape)
                selected_indices = random.sample(noise_parameter_indices_shape, random_num + 1)
                sp_noisy = bfm.add_noise_shape(sp, selected_indices, sigma=sigma_shape)
                random_num = np.random.randint(params_size_exp)
                selected_indices = random.sample(noise_parameter_indices_exp, random_num + 1)
                ep_noisy = bfm.add_noise_exp(ep, selected_indices, sigma=sigma_exp)
                img_from_params(save_folder + 'shape_and_expression', str(i+1) + '_gt', bfm, sp.reshape(199, 1), ep, tp)
                img_from_params(save_folder + 'shape_and_expression', str(i+1) + '_input', bfm, sp_noisy.reshape(199, 1)
                                , ep_noisy.reshape(29, 1), tp)
                out = denoise_shape_and_exp(np.concatenate((sp_noisy, ep_noisy)), net[0], net[1], mode='both')
                out_shape = out[:199, :]
                out_exp = out[199:, :]
                img_from_params(save_folder + 'shape_and_expression', str(i+1) + '_output', bfm,
                                out_shape.reshape(199, 1), out_exp.reshape(29, 1), tp)
                inputs.append((sp_noisy.reshape(199, 1), ep_noisy.reshape(29, 1), tp))
                outputs.append((out_shape.reshape(199, 1), out_exp.reshape(29, 1), tp))
            if i == 5736 + size:
                break
        df = pd.DataFrame(inputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'shape_and_expression/' + 'inputs.pkl')
        df = pd.DataFrame(outputs, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'shape_and_expression/' + 'outputs.pkl')


def generate_noisy_synthetic_data(mode='shape', size=50, save_folder='model_evaluation/synthetic_dataset/dataset/',
                                  range_shape=10, range_exp=10):
    # This function generates a synthetic dataset by choosing 3dmm parameters from a uniform distribution in a large
    # range. So the generated faces are noisy in expression, shape or both due to the given 'mode'.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    labels = []
    for i in range(size):
        if mode == 'shape':
            sp = bfm.get_shape_para_uniform(range=range_shape)
            ep = bfm.get_exp_para('random')
        elif mode == 'exp':
            ep = bfm.get_exp_para_uniform(range=range_exp)
            sp = bfm.get_shape_para('random')
        elif mode == 'both':
            sp = bfm.get_shape_para_uniform(range=range_shape)
            ep = bfm.get_exp_para_uniform(range=range_exp)
        tp = bfm.get_tex_para('random', cov=4)
        img_from_params(save_folder, str(i+1), bfm, sp, ep, tp)
        labels.append((sp, ep, tp))
        df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'labels.pkl')


def denoise_data(net, mode='both', path = 'model_evaluation/synthetic_dataset/dataset/labels.pkl',
                           save_folder='model_evaluation/synthetic_dataset/network_output/', visualize=True):
    # This function takes a .pkl file specified in 'path; containing 3dmm parameters and feeds the parameters to the
    # networks and generates denoised faces in 'save_folder'.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    l = pd.read_pickle(path)
    ep_s = l['ep']
    sp_s = l['sp']
    tp_s = l['tp']
    labels = []
    if mode == 'shape':
        for i, sp in enumerate(sp_s):
            out = denoise_shape_and_exp(sp, net_shape=net, mode='shape')
            ep = ep_s[i]
            tp = tp_s[i]
            if visualize:
                img_from_params(save_folder, str(i + 1), bfm, out, ep, tp)
            labels.append((out, ep, tp))
        df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'outputs.pkl')
    elif mode == 'exp':
        for i, ep in enumerate(ep_s):
            out = denoise_shape_and_exp(ep, net_exp=net, mode='exp')
            sp = sp_s[i]
            tp = tp_s[i]
            if visualize:
                img_from_params(save_folder, str(i+1), bfm, sp, out, tp)
            labels.append((sp, out, tp))
        df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'outputs.pkl')
    elif mode == 'both':
        for i, ep in enumerate(ep_s):
            out = denoise_shape_and_exp(np.concatenate((sp_s[i], ep)), net[0], net[1], mode='both')
            out_s = out[:199, :]
            out_e = out[199:, :]
            tp = tp_s[i]
            if visualize:
                img_from_params(save_folder, str(i + 1), bfm, out_s, out_e, tp)
            labels.append((out_s, out_e, tp))
        df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
        df.to_pickle(save_folder + 'outputs.pkl')
  
# computing numerical error for a list of test data
def model_error(noisy_params, label_params, net, feature_size):
    error = 0
    for i, param in enumerate(noisy_params):
        param = torch.from_numpy(param).float()
        param = param.view(-1, 1, feature_size)
        out = net(param)
        out = out.detach().numpy()
        out = out.reshape(-1)
        l = label_params[i].reshape(-1)
        error += ((out - l) ** 2).mean()
    return error / len(label_params)


# numerical test on whole test datasets
def evaluate_test_data_numerical(net_shape, net_exp):
    test_label_params = np.load('dataset_generation/dataset_shape/test/labels.npy')
    test_noisy_params = np.load('dataset_generation/dataset_shape/test/noisy.npy')
    print('The MSE of shape model on test data is: ', model_error(test_noisy_params, test_label_params, net_shape, 199))
    test_label_params = np.load('dataset_generation/dataset_expression/test/labels.npy')
    test_noisy_params = np.load('dataset_generation/dataset_expression/test/noisy.npy')
    print('The MSE of expression model on test data is: ', model_error(test_noisy_params, test_label_params, net_exp, 29))

def PCA_transform(vectors, all_vectors, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit_transform(all_vectors)
    transform = pca.transform(vectors)
    return transform

# scatter plots
def plot_scatter(mode, net, size, visualize, save_folder, uniform_range):
    if not os.path.exists(save_folder + 'model_output_on_uniform'):
        os.makedirs(save_folder + 'model_output_on_uniform')
    if not os.path.exists(save_folder + 'normal'):
        os.makedirs(save_folder + 'normal')
    # first generating noisy faces by choosing parameters from a uniform distribution
    labels = []
    if mode == 'shape':
        feature_length = 199
        for i in range(size):      
            sp = bfm.get_shape_para_uniform(range=uniform_range)
            ep = bfm.get_exp_para('random')
            tp = bfm.get_tex_para('random', cov=4)
            if visualize:
                img_from_params(save_folder + 'model_output_on_uniform', str(i + 1), bfm, sp, ep, tp)
            labels.append((sp, ep, tp))

    elif mode == 'exp':
        feature_length = 29
        for i in range(size):      
            sp = bfm.get_shape_para('random')
            ep = bfm.get_exp_para_uniform(range=uniform_range)
            tp = bfm.get_tex_para('random', cov=4)
            if visualize:
                img_from_params(save_folder + 'model_output_on_uniform', str(i + 1), bfm, sp, ep, tp)
            labels.append((sp, ep, tp))

    df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
    df.to_pickle(save_folder + 'model_output_on_uniform/' + 'labels.pkl')
    l = pd.read_pickle(save_folder + 'model_output_on_uniform/' + 'labels.pkl')
    ep_s = l['ep']
    sp_s = l['sp']
    tp_s = l['tp']
    labels = []
    if mode == 'shape':
        for i, sp in enumerate(sp_s):
            out = denoise_shape_and_exp(sp, net_shape=net, mode='shape')
            ep = ep_s[i]
            tp = tp_s[i]
            labels.append((sp, ep, tp))
            if visualize:
                img_from_params(save_folder + 'model_output_on_uniform', str(i + 1) + '_out', bfm, out, ep, tp)
    elif mode == 'exp':
        for i, ep in enumerate(ep_s):
            out = denoise_shape_and_exp(ep, net_exp=net, mode='exp')
            sp = sp_s[i]
            tp = tp_s[i]
            labels.append((sp, ep, tp))
            if visualize:
                img_from_params(save_folder + 'model_output_on_uniform', str(i + 1) + '_out', bfm, sp, out, tp)
    df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
    df.to_pickle(save_folder + 'model_output_on_uniform/' + 'labels_out.pkl')
    labels = []
    for i in range(size):
        sp = bfm.get_shape_para('random')
        ep = bfm.get_exp_para('random')
        tp = bfm.get_tex_para('random')
        if visualize:
            img_from_params(save_folder + 'normal', str(i + 1), bfm, sp, ep, tp)
        labels.append((sp, ep, tp))
    df = pd.DataFrame(labels, columns=['sp', 'ep', 'tp'])
    df.to_pickle(save_folder + 'normal/'
                 + 'labels.pkl')
    l1 = pd.read_pickle(save_folder + 'normal/' + 'labels.pkl')
    l2 = pd.read_pickle(save_folder + 'model_output_on_uniform/' + 'labels_out.pkl')
    if mode == 'shape':
        param_normal = l1['sp']
        param_uniform = l2['sp']
    elif mode == 'exp':
        param_normal = l1['ep']
        param_uniform = l2['ep']

    param_uniform = np.array(param_uniform)
    param_u = np.zeros((size, feature_length))
    for i in range(size):
        param_u[i, :] = param_uniform[i].reshape(-1)
    param_u = param_u.reshape(-1, feature_length)
    param_normal = np.array(param_normal)
    param_n = np.zeros((size, feature_length))
    for i in range(size):
        param_n[i, :] = param_normal[i].reshape(-1)
    param_n = param_n.reshape(-1, feature_length)
    
    src_dir = 'dataset_generation/AFLW2000_AND_300W/'
    param_real = []
    if mode == 'shape':
        for i, file in enumerate(os.listdir(src_dir)):
            if i < size:
                mat = sio.loadmat(src_dir + file)
                sp = mat['Shape_Para']
                param_real.append(sp)
    elif mode == 'exp':
        for i, file in enumerate(os.listdir(src_dir)):
            if i < size:
                mat = sio.loadmat(src_dir + file)
                ep = mat['Exp_Para']
                param_real.append(ep)   
    param_real = np.array(param_real)
    param_real = param_real.reshape(-1, feature_length)

    all_vectors = np.concatenate((param_n, param_u, param_real))
    transform_n = PCA_transform(param_n, all_vectors, n_components=2)
    transform_u = PCA_transform(param_u, all_vectors, n_components=2)
    transform_r = PCA_transform(param_real, all_vectors, n_components=2)


    plt.figure()
    s = 30
    plt.scatter(transform_n[:, 0], transform_n[:, 1], s=s, marker='*')
    plt.scatter(transform_r[:, 0], transform_r[:, 1], s=s, marker='o')
    plt.scatter(transform_u[:, 0], transform_u[:, 1], s=s, marker='^')
    plt.legend(('normal distribution', 'fitted parameters to real images', 'output of the network on uniform distribution'))
    plt.savefig(save_folder + 'fig' + str(size) + '.png', dpi=400)
    plt.show()
    
    cov_u = np.cov(param_u.transpose())
    trace_u = np.trace(cov_u)
    
    cov_n = np.cov(param_n.transpose())
    trace_n = np.trace(cov_n)
    
    cov_r = np.cov(param_real.transpose())
    trace_r = np.trace(cov_r)
    
    print('Normal: ', trace_n , ' Realistic: ', trace_r , ' Uniform: ', trace_u)
    
