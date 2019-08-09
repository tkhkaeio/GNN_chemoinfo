import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error")

def load_model(model, args):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load("experiment/{}/model/{}"(args.name, args.resume))
            start_epoch = checkpoint['epoch']
            rand_state = checkpoint['rand_state']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            start_epoch = 0
            rand_state = args.random_state
    return start_epoch, rand_state

def multi_process(model, args):
    if torch.cuda.device_count() > 1:
        if args.gpu is not None:
            print("Use", args.gpu, "GPUs")
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            print("Use", torch.cuda.device_count(), "GPUs")
            ngpus_per_node = torch.cuda.device_count()
            model = torch.nn.DataParallel(model)
            print("Data was distributed")
    return model

def trapezoid(x1, x2, y1, y2):
    width = abs(x2-x1)
    height = (y1 + y2) / 2.
    return width * height

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(losses_train, losses_valid, args, rand_state=None):
    assert len(losses_train) == len(losses_valid)
    plt.clf()
    plt.plot(losses_train, label="train")
    plt.plot(losses_valid, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim((0, max(max(losses_train), max(losses_valid))*1.1))
    plt.legend(loc = "upper right")
    plt.savefig("experiments/{}/loss_{}.png".format(args.name, rand_state))

def confusion_matrix(ps, labels, thresh_hold=0.5, mode=None, epoch=0, args=None, rand_state=None):
    best_result = (0,[0,0,0,0,0]); auc = 0; y_preds = []
    ConfusionMatrix = np.zeros((2,2))
    for i,p in enumerate(ps):
        y_pred = 1 if p > thresh_hold else 0
        if(len(labels)==0): label = y_pred
        else: label = labels[i]
        y_preds.append(y_pred)
        ConfusionMatrix[1-y_pred][1-int(label)] += 1
    acc = (ConfusionMatrix[0][0] + ConfusionMatrix[1][1])/np.sum(ConfusionMatrix)
    try: precision = ConfusionMatrix[0][0]/(np.sum(ConfusionMatrix, axis=1)[0])
    except: precision = 0
    try: recall = ConfusionMatrix[0][0]/(np.sum(ConfusionMatrix, axis=0)[0])
    except: recall = 0
    try: FPRscores = ConfusionMatrix[0][1]/(ConfusionMatrix[0][1] + ConfusionMatrix[1][0])
    except: FPRscores = 0
    try: F = 2*precision*recall/(precision+recall)
    except: F = 0

    if mode == "ROC":
        pred_labels = []
        for i,j in zip(ps, labels): pred_labels.append((i,j))
        pred_labels = sorted(pred_labels, key=lambda tuple: -tuple[0])
        ps = []; labels = []
        for i,j in pred_labels:
            ps.append(i); labels.append(j)
        
        prev_y = 0; prev_x = 0
        recall_list = []; FPRscores_list = []
        threses = np.arange(0,1, 0.050)[::-1]
        for thresh in threses:
            scores ,_, _ = confusion_matrix(ps, labels, thresh_hold=thresh)
            #print("eval ROC: thresh_hold: {:.3f}, valid_recall: {:.3f}, valid_FPR: {:.3f}".format(thresh, scores[2], scores[3]))
            x = scores[3]; y = scores[2]
            auc += trapezoid(prev_x, x, prev_y, y)
            FPRscores_list.append(x)
            recall_list.append(y)
            if((y-x) > (prev_y-prev_x)): best_result = (thresh, scores)
            prev_x = x; prev_y = y
        
        if epoch%10 == 0 or epoch==args.epochs-1:
            plt.xlabel("FPR")
            plt.ylabel("recall")
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.plot([0,1], [0,1], linestyle='dashed', alpha=0.7)
            plt.plot(FPRscores_list, recall_list, label="ROC_{} AUC={:.3f} F(thresh=0.5)={:.3f}".format(epoch, auc, F))
            plt.legend(loc = "lower right")
            plt.savefig("experiments/{}/ROC_{}.png".format(args.name, rand_state))

        print("ConfusionMatrix\n", ConfusionMatrix)
    return [acc, precision, recall, FPRscores, F], y_preds, best_result


