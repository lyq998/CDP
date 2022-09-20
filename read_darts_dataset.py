import os
import torch
from cross_domain_predictor import GCN_predictor, get_train_dataloader, get_target_train_dataloader
import argparse
from scipy.stats import kendalltau
import numpy as np
from torch.utils.data import DataLoader
from dataset_matrix import Dataset_Darts
import matplotlib.pyplot as plt
import pickle

filename = os.path.join('tiny_darts', 'darts_dataset.pth.tar')
data = torch.load(filename)

parser = argparse.ArgumentParser(description='darts_test')
parser.add_argument('--integers2one_hot', type=bool, default=True, help='whether to transform integers -> one_hot')
parser.add_argument('--train_batch_size', default=500, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--seed', type=int, default=6, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--which_assistant', type=int, default=0, help='0 for tiny, 1 for small tiny')
parser.add_argument('--ns', default=True, type=bool, help='whether to forbidden skip in assistant space')
parser.add_argument('--increasing_speed', default='cos', type=str, help='cos: faster, linear: const, sin: slower')
parser.add_argument('-K', '--max_subdomains', default=3, type=int, help='the max number of subdomains')
parser.add_argument('--kernel_type', default='rbf', type=str, help='choice from rbf, laplace, and rqk')
parser.add_argument('--using_dataset', default='all', type=str, choices=['all', '101', '201'])
parser.add_argument('--show_figure', default=False, type=bool)
parser.add_argument('--top_k', default=True, type=bool)
parser.add_argument('--figure_index', default=2, type=int, help='the index of the saving figure')
# parser.add_argument('--predictor', type=str, default='GCN', choices=['RF', 'GCN'])
parser.add_argument('--loss_type', default='lmmd', type=str, help='lmmd, coral')
parser.add_argument('--is_adv', default=False, type=bool, help='Whether using adversarial loss')
args = parser.parse_args()

if __name__ == '__main__':
    normal_layers = 6

    print("args =", args)

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataloader, percentile = get_train_dataloader(normal_layer=normal_layers,
                                                        train_batch_size=args.train_batch_size,
                                                        percentile=True, using_dataset=args.using_dataset)
    target_dataloader = get_target_train_dataloader(args.train_batch_size, dataset_num=len(data['dataset']),
                                                    dataset=data['dataset'])
    # choose which assistant space to use
    if args.which_assistant == 0:
        dataset_type = 'tiny'
    else:
        dataset_type = 'small_tiny'
    Tiny_darts = Dataset_Darts(dataset_num=len(data['dataset']), dataset_type=dataset_type, ns=args.ns)
    assistant_dataloader = DataLoader(Tiny_darts, batch_size=args.train_batch_size, shuffle=True)

    K = args.max_subdomains
    predictor = GCN_predictor(percentile, speed=args.increasing_speed, K=K, is_adv=args.is_adv)
    # add assistant_dataloader
    predictor.train(train_dataloader, target_dataloader, assistant_dataloader, loss_type=args.loss_type,
                    kernel_type=args.kernel_type)

    Darts_Matrix = Dataset_Darts(dataset_num=len(data['dataset']), dataset=data['dataset'])
    dataloader_darts = DataLoader(Darts_Matrix, batch_size=args.test_batch_size, shuffle=False)
    pred_y = predictor.predict(dataloader_darts, normal_layer=normal_layers)

    true_y = np.array(data['best_acc_list'])
    # test top-k architectures
    if args.top_k:
        # acsend
        indexs = np.argsort(true_y)
        for k in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]:
            top_indexs = indexs[-k:]
            top_true_y, top_pred_y = [], []
            for i in top_indexs:
                top_true_y.append(true_y[i])
                top_pred_y.append(pred_y[i])
            print('Top {} KTau: {}'.format(k, kendalltau(top_pred_y, top_true_y)[0]))
        
    print('KTau: {}'.format(kendalltau(pred_y, true_y)[0]))
    # print('MSE: {}'.format(mean_squared_error(pred_y, ture_y) / 10000))

    # save
    # data = {'pred_y': pred_y, 'true_y': true_y}
    #
    # with open('save.pkl', 'wb') as file:
    #     pickle.dump(data, file)

    if args.show_figure:
        pred_rank = np.argsort(np.argsort(pred_y))
        ture_rank = np.argsort(np.argsort(true_y))

        x = np.arange(0, len(true_y), 0.1)
        y = x
        plt.figure(figsize=(3, 3))
        line_color = '#1F77D0'
        plt.plot(x, y, c=line_color, linewidth=1.5)
        point_color = '#FF4400'
        plt.scatter(pred_rank, ture_rank, c=point_color, s=4)
        # plt.xlabel("predict_result")
        # plt.ylabel("y_test")
        plt.xlim(xmax=100, xmin=0)
        plt.ylim(ymax=100, ymin=0)
        # plt.show()
        plt.savefig(os.path.join('fig', 'KTau{}.pdf'.format(args.figure_index)))
