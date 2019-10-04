# Face Manifold
By [Kimia Dinashi](https://github.com/dinashi) and [Ramin Toosi](https://github.com/ramintoosi)


## Introduction
This repository is the implementation of the paper entitled "[Face Manifold: Manifold Learning for Synthetic Face Generation](http://arxiv.org/abs/1910.01403)". This could be used for generating synthetic 3D and 2D face datasets with large variations in face shape and expression.

## Getting Started

### Requirements
- Platform: Linux (Windows is not tested.)
- Python 3
- Pytorch = 1.0.1
- Numpy = 1.15.2
- Pandas = 0.24.2
- Scipy = 1.1.0
- Skimage = 0.14.2
- Sklearn = 0.20.3

### Usage

1. Download 3D morphabel model and prepare BFM.mat as explained in https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md . Then put it in `/utils_for_3dmm` folder. 

2. Download 300W-3D and AFLW2000-3D 3DMM databases from http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm . Copy all the `.mat` files in the directory `/dataset_generation/AFLW2000_AND_300W/`.

#### Testing
Run **test.py** to test the model performance. A Gaussian noise would be added to the 3DMM samples of the dataset exisiting in `/dataset_generation/AFLW2000_AND_300W/`. The noisy samples are fed to the networks for denoising. The results would be saved in `/model_evaluation/output_on_test_data/`. The arguments are:

  1) mode: to determine that the noise is only added to the expression, or the shape parameters, or both.
  2) size: how many of the dataset samples are used for testing.
  3) sigma_shape and sigma_exp: standard deviation of the Gaussian noise added to the shape and expression parameters, respectively.
  
 `python test.py -mode shape -size 20 -sigma_shape 100000 -sigma_exp 1`

Run **plot_scatter_diagram.py** to plot scatter diagrams representing diversity level of the generated faces. The generated faces with our method are compared to the generated faces using normal distribution and also the realistic faces existing in the dataset. The arguments are:
 1) mode: determines which scatter diagram to plot: shape, expression or both.
 2) size: determines the number of samples used to plot the scattering diagram.
 3) visualize: if True, the 2D image of generated faces would be saved in 'model_evaluation/scatter_diagrams/'.
 4) range: the degree of diversity in the generated dataset.
 
 `python plot_scatter_diagram -mode exp -size 100 -visualize False -range 10`

#### Synthetic Data Generation
Run **create_synthetic_dataset.py** for creating synthetic 3D and 2D face dataset. First a synthetic dataset by choosing 3DMM parameters from a uniform distribution in a large range (high diversity) is created. The generated faces are noisy in expression, shape or both due to the given `mode`. Then the noisy 3DMM parameters are fed to the networks and the denoised version of them would be saved in `/model_evaluation/synthetic_data/network_output`. The arguments are:
  1) mode: to determine which parameters are chosen from a uniform distribution, shape, expression or both.
  2) size: size of the generated dataset
  3) shape_range and exp_range: the degree of diversity in the facial shapes and expressions, respectively.
  
  `python create_synthetic_dataset.py -mode both -size 50 -shape_range 10 -exp_range 10`
 
Run **denoise.py** to denoise 3DMM parameters stored as a pandas dataframe in a .pkl file. The keys are 'sp', 'ep' and 'tp' (see [inputs.pkl](https://github.com/dinashi/Denoising_3D_Face/blob/master/model_evaluation/output_on_test_data/expression/inputs.pkl) as an example). The arguments are:
1) mode: for choosing between shape, expression or both to denoise.
2) path: path to the .pkl input file.
3) save_folder: where the output pkl file would be saved.
4) visualize: if True, the image of generated faces would be saved the in save_folder.

` python denoise.py -mode exp -path ./model_evaluation/output_on_test_data/expression/inputs.pkl -save_folder ./model_evaluation/output_on_test_data/expression/ -visualize True`

 
#### Training
1) Run **generate_shape_dataset.py** and **generate_expression_dataset.py** to generate datasets.
2) Run **train_shape.py** and **train_expression.py** to train shape and expression networks respectively. The trained models would be saved in trained_models directory.

## Acknowledgement

Thanks to [YadiraF](https://github.com/YadiraF) for providing python tools for processing 3D face. The main part of the code in utils_for_3dmm directory is used from the repository https://github.com/YadiraF/face3d.

  

