import os
import torch
from cross_domain_predictor import GCN_predictor
import argparse
from scipy.stats import kendalltau
import numpy as np
from torch.utils.data import DataLoader
from dataset_matrix import Dataset_Darts
import matplotlib.pyplot as plt
import pickle
from torch import optim
from utils import get_logger, to_cuda, convert_to_genotype
from torch import nn

filename = os.path.join('tiny_darts', 'darts_dataset.pth.tar')
train_data = torch.load(os.path.join('path', 'darts_dataset_fullytrain.pth.tar'))
data = torch.load(filename)

k=50
train_data['dataset'] = train_data['dataset'][:k]
train_data['best_acc_list'] = train_data['best_acc_list'][:k]

# train_data['dataset'] = data['dataset'][:k]
# train_data['best_acc_list'] = data['best_acc_list'][:k]

# data['dataset'] = data['dataset'][k:]
# data['best_acc_list'] = data['best_acc_list'][k:]

# # 1,2 <-> 3,4 op
# for geno in train_data['dataset']:
#     for i, g in enumerate(geno):
#         for j, op in enumerate(g):
#             pre= op[0]
#             if op[1] ==1:
#                 geno[i][j] = (pre, 3)
#             if op[1] ==2:
#                 geno[i][j] = (pre, 4)
#             if op[1] ==3:
#                 geno[i][j] = (pre, 1)
#             if op[1] ==4:
                # geno[i][j] = (pre, 2)

parser = argparse.ArgumentParser(description='darts_test')
parser.add_argument('--integers2one_hot', type=bool, default=True, help='whether to transform integers -> one_hot')
parser.add_argument('--train_batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--which_assistant', type=int, default=0, help='0 for tiny, 1 for small tiny')
parser.add_argument('--ns', default=True, type=bool, help='whether to forbidden skip in assistant space')
parser.add_argument('--show_figure', default=False, type=bool)
parser.add_argument('--figure_index', default=2, type=int, help='the index of the saving figure')
# parser.add_argument('--predictor', type=str, default='GCN', choices=['RF', 'GCN'])
args = parser.parse_args()

if __name__ == '__main__':
    normal_layers = 6

    print("args =", args)

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Darts_Matrix = Dataset_Darts(dataset_num=len(train_data['dataset']), dataset=train_data['dataset'])
    dataloader_darts = DataLoader(Darts_Matrix, batch_size=args.test_batch_size, shuffle=False)

    predictor = GCN_predictor(None)
    predictor.train_without_domain_adaptation(dataloader_darts, train_data['best_acc_list'])

    Darts_Matrix = Dataset_Darts(dataset_num=len(data['dataset']), dataset=data['dataset'])
    dataloader_darts = DataLoader(Darts_Matrix, batch_size=args.test_batch_size, shuffle=False)
    pred_y = predictor.predict(dataloader_darts, normal_layer=normal_layers)

    true_y = np.array(data['best_acc_list'])
        
    print('KTau: {}'.format(kendalltau(pred_y, true_y)[0]))