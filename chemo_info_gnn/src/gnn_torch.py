import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from data import *
from utils import *
from function import *
import time
import os

def args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',           default="baseline",  type=str,   help='to name the model')
    parser.add_argument('--lr',             default=0.0001,      type=float, help='to set the parameters')
    parser.add_argument('--weight-decay',   default=1e-4,        type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--aggregate',      default="sum",       type=str,   help='function in aggregate (sum|mean|max)')
    parser.add_argument('--activation',     default="ReLU",      type=str,   help='name of activation (ReLU|LeakyReLU)')
    parser.add_argument('--MLP_layer',      default=1,           type=int,   help='num of MLP layers')
    parser.add_argument('--optimizer',      default="SGD",       type=str,   help='optimizer (SGD|momentum_SGD')
    parser.add_argument('--epochs',          default=20,          type=int,   help='num of epoch')
    parser.add_argument('--random_state',   default=0,           type=int,   help='num of random state')
    parser.add_argument('--num_rand_sample',default=1,           type=int,   help='num of random ranple')
    parser.add_argument('--batch_size',     default=200,         type=int,   help='num of batch size')
    parser.add_argument('--mean',           default=0,           type=float, help='mean of init')
    parser.add_argument('--std',            default=0.4,         type=float, help='std of init')
    parser.add_argument('--X_dim',          default=8,           type=int,   help='dim of feature vector in graph')
    parser.add_argument('--step',           default=2,           type=int,   help='num of aggregate step')
    parser.add_argument('--eta',            default=0.8,         type=float, help='params of eta in momentum SGD')
    parser.add_argument('--num_workers',    default=4,           type=int,   help='num of workers in load')
    parser.add_argument('--tox21',          default=True,        type=bool,  help='whetehr to load tox21 data')
    parser.add_argument('--resume',         default="model.pth", type=str,   help='name of a load model')
    parser.add_argument('--gpu',            default=None,        type=int,   help='id of gpus to use')

    return parser.parse_args()

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.linear1.weight, args.mean, args.std)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x = self.linear1(x)
        p = torch.sigmoid(x)
        return p, x

class GraphNeuralNetework(nn.Module):
    def __init__(self, args):
        super(GraphNeuralNetework, self).__init__()
        self.step = args.step
        self.X_dim = args.X_dim
        self.aggregate = args.aggregate
        self.activation = args.activation
        self.MLP_layer = args.MLP_layer
        if self.MLP_layer == 1:
            self.transform = Linear(self.X_dim, self.X_dim, args).to(device)
        else: 
            self.transform = MLP(self.X_dim, self.X_dim, self.X_dim, args).to(device) 
        self.classifier = Classifier(self.X_dim, 1, args).to(device)
    
    def init_hidden(self, X, batch_data):
        for i, adj_list in enumerate(batch_data):
            if i >= len(X): break
            for j in range(adj_list.node_num): X[i][j][0] = 1
    
    def AGGREGATE_COMBINE(self, X, batch_data):
        X_next = torch.zeros_like(X, dtype=torch.float).to(device)
        for i, adj_list in enumerate(batch_data):
            if i >= len(X): break
            neighbors = adj_list.neighbors
            for j in range(adj_list.node_num):
                a_list = []
                for k in neighbors[j]: a_list.append(X[i][k].detach().cpu().numpy()) #node j <- neighbor k
                if len(a_list) == 0: continue
                elif len(a_list) == 1: pass
                elif self.aggregate=="sum":  a = torch.FloatTensor(a_list).sum(dim=0).to(device)
                elif self.aggregate=="mean": a = torch.FloatTensor(a_list).mean(dim=0).to(device)
                elif self.aggregate=="max":  a = torch.FloatTensor(a_list).max(dim=0)[0].to(device)
                else: print("cannot locate aggregation function", self.aggregate); exit()
                X_next[i][j] = self.transform(a) #f(W*a)
        return X_next
    
    def READOUT(self, X):
        return torch.sum(X, dim=1)

    def forward(self, X, batch_data, params=None, train=True, debug=False):
        self.init_hidden(X, batch_data)
        for _ in range(self.step):
            X = self.AGGREGATE_COMBINE(X, batch_data)
        Hg = self.READOUT(X)
        Hg_check(Hg, args)
        p, out = self.classifier(Hg)
        return p, out

def main():
    os.makedirs("experiments/"+args.name, exist_ok=True); os.makedirs("experiments/{}/model".format(args.name), exist_ok=True)

    if args.tox21:
        print("load tox 21")
        adj_list_train, adj_list_valid = tox21_loader()
        train_data = MyDataLoader(adj_list_train, args.batch_size,  shuffle=True, num_workers=args.num_workers)
        valid_data = MyDataLoader(adj_list_valid, batch_size=None, shuffle=False, num_workers=args.num_workers)
    else:
        adj_list_train, adj_list_test = load_data()
        print("train: {}, valid: {}, test: {}".format(int(len(adj_list_train)*0.8), int(len(adj_list_train)*0.2), len(adj_list_test)))
        train_data = MyDataLoader(adj_list_train[0:int(len(adj_list_train)*0.8)], args.batch_size,  num_workers=args.num_workers, shuffle=True) 
        valid_data = MyDataLoader(adj_list_train[int(len(adj_list_train)*0.8):], batch_size=None,  num_workers=args.num_workers, shuffle=False)
        test_data = MyDataLoader(adj_list_test, batch_size=None,  num_workers=args.num_workers, shuffle=False)

    criterion = nn.BCEWithLogitsLoss().to(device)
    ACC_train = []; ACC_valid = []; F_train = []; F_valid = []; BEST_result = (0,[0,0,0,0,0]); BEST_epoch=0
    start = time.time()
    for _ in range(args.num_rand_sample):
        plt.close()
        gnn = GraphNeuralNetework(args)
        start_epoch, rand_state = load_model(gnn, args)
        elapsed_time = time.time() - start
        print("{} h {} min elapsed, start learning from: epoch {}, random state {}".format(elapsed_time//60*60, elapsed_time//60, start_epoch, rand_state), "\n")
        np.random.seed(rand_state); torch.manual_seed(rand_state)
        cudnn.deterministic = True; cudnn.benchmark=True
        gnn = multi_process(gnn, args).to(device)
        print("params: ", count_parameters(gnn))

        if args.optimizer == "SGD": optimizer = optim.SGD(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "momentum_SGD": optimizer = optim.SGD(gnn.parameters(), lr=args.lr, momentum=args.eta, weight_decay=args.weight_decay)

        losses_train_log = []; losses_valid_log = []
        for epoch in range(start_epoch, args.epochs): 
            losses_train = []; losses_valid = []; acc_train = []; acc_valid = []; f_train = []; f_valid = []

            gnn.train()
            for _,labels,batch_data in train_data:
                max_dim = max([adj_list.node_num for adj_list in batch_data])

                gnn.zero_grad()
                labels = torch.FloatTensor(labels).to(device)
                X = torch.zeros((args.batch_size, max_dim, args.X_dim), dtype=torch.float).to(device)
                y, outs = gnn.forward(X, batch_data)

                loss = criterion(outs, labels.unsqueeze(1)) 
                scores,_,_ = confusion_matrix(y.squeeze(1), labels, epoch=epoch, args=args, rand_state=rand_state)
                
                loss.backward()
                optimizer.step()

                losses_train.append(loss.tolist())
                acc_train.append(scores[0])
                f_train.append(scores[4])
            
            gnn.eval()
            for _,labels,batch_data in valid_data:
                max_dim = max([adj_list.node_num for adj_list in batch_data])

                labels = torch.FloatTensor(labels).to(device)
                X = torch.zeros((len(batch_data), max_dim, args.X_dim), dtype=torch.float).to(device)
                y, outs = gnn.forward(X, batch_data)

                loss = criterion(outs, labels.unsqueeze(1)) 
                scores,_, best_result = confusion_matrix(y.squeeze(1), labels, mode="ROC", epoch=epoch, args=args, rand_state=rand_state)

                losses_valid.append(loss.tolist())
                acc_valid.append(scores[0])
                f_valid.append(scores[4])
            
            losses_train_log.append(np.mean(losses_train)); losses_valid_log.append(np.mean(losses_valid))
            print('EPOCH: {}, Train(thresh=0.5) [Loss: {:.3f}, acc: {:.3f}, F: {:.3f}], Valid(thresh=0.5) [Loss: {:.3f}, acc: {:.3f}, F: {:.3f}]'.format(
                epoch,
                losses_train_log[epoch],
                np.mean(acc_train),
                np.mean(f_train),
                losses_valid_log[epoch],
                np.mean(acc_valid),
                np.mean(f_valid)
            ))
            print("EPOCH: {}, best thresh hold: {:.3f}, valid acc: {:.3f}, valid F: {:.3f}".format(
                epoch,
                best_result[0],
                best_result[1][0],
                best_result[1][4]
            ))
            if best_result[1][4] > BEST_result[1][4]:
                BEST_epoch = epoch
                BEST_result = best_result
        #save model
        torch.save({
            'epoch': epoch,
            'rand': rand_state,
            'model_state_dict': gnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "experiments/{}/model/model_{}.pth".format(args.name, rand_state))
        plot_loss(losses_train_log, losses_valid_log, args, rand_state=rand_state)
        ACC_train.append(acc_train[-1]); ACC_valid.append(acc_valid[-1]); F_train.append(f_train[-1]); F_valid.append(f_valid[-1])
        print("\n\nrand_state: {}, epoch: {}".format(rand_state, epoch))
        print("result train: mean ACC:{mean:.3f}\pm{diff:.3f}, max ACC:{max:.3f}, min ACC:{min:.3f}, diff:{diff:.3f}".format(mean=np.mean(ACC_train), max=max(ACC_train), min=min(ACC_train), diff=(max(ACC_train)-np.mean(ACC_train))))
        print("result train: mean   F:{mean:.3f}\pm{diff:.3f}, max   F:{max:.3f}, min   F:{min:.3f}, diff:{diff:.3f}".format(mean=np.mean(F_train), max=max(F_train), min=min(F_train), diff=(max(F_train)-np.mean(F_train))))
        print("result valid: mean ACC:{mean:.3f}\pm{diff:.3f}, max ACC:{max:.3f}, min ACC:{min:.3f}, diff:{diff:.3f}".format(mean=np.mean(ACC_valid), max=max(ACC_valid), min=min(ACC_valid),diff=(max(ACC_valid)-np.mean(ACC_valid))))
        print("result valid: mean   F:{mean:.3f}\pm{diff:.3f}, max   F:{max:.3f}, min   F:{min:.3f}, diff:{diff:.3f}".format(mean=np.mean(F_valid), max=max(F_valid), min=min(F_valid), diff=(max(F_valid)-np.mean(F_valid))))
        print("BEST result EPOCH: {}, best thresh hold: {:.3f}, valid acc: {:.3f}, valid F: {:.3f}".format(
            BEST_epoch,
            BEST_result[0],
            BEST_result[1][0],
            BEST_result[1][4]
        ))
        args.random_state += 1

if __name__ == '__main__':
    args = args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()