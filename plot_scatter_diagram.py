import torch
import argparse
from training.network import AutoEncoder4
from model_evaluation.funcs import plot_scatter

parser = argparse.ArgumentParser()
parser.add_argument("-mode", default="both", help="mode can be shape, exp, or both.")
parser.add_argument("-size", default=200, help="number of generated faces.", type=int)
parser.add_argument("-visualize", default=False, help="if visualize is true, the face images are created.", type=bool)
parser.add_argument("-range", default=10, help="range is multiplied by the range of the uniform distribution", type=int)
args = parser.parse_args()

net_shape = AutoEncoder4()
net_shape.load_state_dict(torch.load('./trained_models/shape/epoch_10.pkl', map_location='cpu'))
net_exp = AutoEncoder4()
net_exp.load_state_dict(torch.load('./trained_models/expression/epoch_10.pkl', map_location='cpu'))

# scatter diagrams
save_folder_shape = 'model_evaluation/scatter_diagrams/shape/'
save_folder_exp = 'model_evaluation/scatter_diagrams/expression/'
if args.mode == 'shape':
    print('Plotting scatter diagram for shape parameters...')
    plot_scatter('shape', net_shape, args.size, args.visualize, save_folder_shape, args.range)
elif args.mode == 'exp':
    print('Plotting scatter diagram for expression parameters...')
    plot_scatter('exp', net_exp, args.size, args.visualize, save_folder_exp, args.range)
elif args.mode == 'both':
    print('Plotting scatter diagram for shape parameters...')
    plot_scatter('shape', net_shape, args.size, args.visualize, save_folder_shape, args.range)
    print('Plotting scatter diagram for expression parameters...')
    plot_scatter('exp', net_exp, args.size, args.visualize, save_folder_exp, args.range)
    
