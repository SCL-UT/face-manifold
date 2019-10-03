import torch
import argparse
from training.network import AutoEncoder4
from model_evaluation.funcs import evaluate_test_data

parser = argparse.ArgumentParser()
parser.add_argument("-mode", default="shape", help="mode can be shape, exp, or both.")
parser.add_argument("-size", help="number of face images for testing.", default=20, type=int)
parser.add_argument("-sigma_shape", default=100000, help="sigma determines SD of the Gaussian noise.", type=int)
parser.add_argument("-sigma_exp", default=1, help="sigma determines SD of the Gaussian noise.", type=int)

args = parser.parse_args()
net_shape = AutoEncoder4()
net_shape.load_state_dict(torch.load('./trained_models/shape/epoch_10.pkl', map_location='cpu'))
net_exp = AutoEncoder4()
net_exp.load_state_dict(torch.load('./trained_models/expression/epoch_10.pkl', map_location='cpu'))

# qualitative results
if args.mode == 'shape':
    evaluate_test_data(net_shape, args.size, sigma_shape=args.sigma_shape, sigma_exp=args.sigma_exp, mode='shape')
elif args.mode == 'exp':
    evaluate_test_data(net_exp, args.size, sigma_shape=args.sigma_shape, sigma_exp=args.sigma_exp, mode='exp')
elif args.mode == 'both':
    evaluate_test_data((net_shape, net_exp), args.size, sigma_shape=args.sigma_shape, sigma_exp=args.sigma_exp, mode='both')
else:
    print("mode can be shape, exp, or both.")
