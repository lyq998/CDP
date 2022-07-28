import os
import torch
from cross_domain_predictor import NeuralPredictor
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

    # torch.cuda.set_device(args.gpu)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    predictor = NeuralPredictor()
    epochs = 50
    optimizer = optim.Adam(predictor.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        batch = to_cuda(train_data)
        s_label = batch["val_acc"].to(torch.float)

        predict = predictor(batch.dataset)
        optimizer.zero_grad()
        loss = criterion(predict, s_label)
        
        loss.backward()
        optimizer.step()

    Darts_Matrix = Dataset_Darts(dataset_num=len(data['dataset']), dataset=data['dataset'])
    dataloader_darts = DataLoader(Darts_Matrix, batch_size=args.test_batch_size, shuffle=False)
    pred_y = predictor.predict(dataloader_darts, normal_layer=normal_layers)

    true_y = np.array(data['best_acc_list'])
        
    print('KTau: {}'.format(kendalltau(pred_y, true_y)[0]))