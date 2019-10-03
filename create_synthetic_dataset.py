import torch
import argparse
from training.network import AutoEncoder4
from model_evaluation.funcs import generate_noisy_synthetic_data, denoise_data

parser = argparse.ArgumentParser()
parser.add_argument("-mode", help="mode can be shape, exp, or both.", default="both")
parser.add_argument("-size", help="size of the generated dataset.", default=50, type=int)
parser.add_argument("-shape_range", help="range determines the degree of diversity in the face shapes.", default=10, type=int)
parser.add_argument("-exp_range", help="range determines the degree of diversity in the face expressions.", default=
                    10, type=int)

args = parser.parse_args()
net_shape = AutoEncoder4()
net_shape.load_state_dict(torch.load('./trained_models/shape/epoch_10.pkl', map_location='cpu'))
net_exp = AutoEncoder4()
net_exp.load_state_dict(torch.load('./trained_models/expression/epoch_10.pkl', map_location='cpu'))

print('Generating noisy faces ...')
generate_noisy_synthetic_data(mode=args.mode, size=args.size, range_shape=args.shape_range, range_exp=args.exp_range)

print('Feeding noisy faces to the network to generate final dataset ...')

# denoising generated dataset
if args.mode == 'shape':
    denoise_data(net_shape, mode=args.mode)
elif args.mode == 'exp':
    denoise_data(net_exp, mode=args.mode)
elif args.mode == 'both':
    denoise_data((net_shape, net_exp), mode=args.mode)
else:
    print("mode can be shape, exp, or both.")




