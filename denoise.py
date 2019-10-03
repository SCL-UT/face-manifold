import torch
import argparse
from training.network import AutoEncoder4
from model_evaluation.funcs import denoise_data

parser = argparse.ArgumentParser()
parser.add_argument("-mode", default="shape", help="mode can be shape, exp, or both.")
parser.add_argument("-path", help="path to the .pkl file containing 3DMM parameters")
parser.add_argument("-save_folder", help="save folder for network outputs.")
parser.add_argument("-visualize", default=False, help="if visualize is true, the face images are created.", type=bool)

args = parser.parse_args()
net_shape = AutoEncoder4()
net_shape.load_state_dict(torch.load('./trained_models/shape/epoch_10.pkl', map_location='cpu'))
net_exp = AutoEncoder4()
net_exp.load_state_dict(torch.load('./trained_models/expression/epoch_10.pkl', map_location='cpu'))

# qualitative results
if args.mode == 'shape':
    denoise_data(net_shape, mode='shape', path=args.path, save_folder=args.save_folder, visualize=args.visualize)
elif args.mode == 'exp':
    denoise_data(net_exp, mode='exp', path=args.path, save_folder=args.save_folder, visualize=args.visualize)
elif args.mode == 'both':
    denoise_data((net_shape, net_exp), mode='both', path=args.path, save_folder=args.save_folder,
                 visualize=args.visualize)
else:
    print("mode can be shape, exp, or both.")
