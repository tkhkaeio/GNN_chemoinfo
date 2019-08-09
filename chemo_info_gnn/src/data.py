import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, utils

class Adjacency_List: 
    def __init__(self, node_num, adj_list, label):
        self.node_num = node_num
        self.data = adj_list
        self.edge_num = len(adj_list)
        self.label = label
        self.neighbors = [[] for i in range(self.node_num)] #neighbors in each vertex
        for i, j in adj_list:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

    def __getitem__(self, item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)
                
def load_data():
    adj_list_train = []
    adj_list_test = []
    label_list_train = []
    maxnum_data = 2000
    
    for i in range(maxnum_data):
        with open("datasets/train/{}_label.txt".format(i), 'r') as f:#, open("../datasets/test/{}_label.txt".format(i), 'r') as f2:
            label_list_train.append(int(f.readline()))
        try:
            nodes_train = []; nodes_test = []
            with open("datasets/train/{}_graph.txt".format(i), 'r') as f, open("datasets/test/{}_graph.txt".format(i), 'r') as f2:
                dims = [(f,int(f.readline().strip())), (f2,int(f2.readline().strip()))]
                for k, (file,dim) in enumerate(dims):
                    for j in range(dim):
                        row = file.readline().strip().split()
                        if k: nodes_test.extend([(j,index) for (index, node) in enumerate(row) if int(node)==1 and (index>j)])
                        else: nodes_train.extend([(j,index) for (index, node) in enumerate(row) if int(node)==1 and (index>j)])
                    if k: adj_list_test.append(Adjacency_List(node_num=dim, adj_list=nodes_test, label=None))
                    else: adj_list_train.append(Adjacency_List(node_num=dim, adj_list=nodes_train, label=label_list_train[i]))
        except FileNotFoundError: #collect the other train data
            with open("datasets/train/{}_graph.txt".format(i), 'r') as file:
                dim = int(file.readline().strip())
                for j in range(dim):
                    row = file.readline().strip().split()
                    nodes_train.extend([(j,index) for (index, node) in enumerate(row) if int(node)==1 and (index>j)])
                adj_list_train.append(Adjacency_List(node_num=dim, adj_list=nodes_train, label=label_list_train[i]))
        
    return adj_list_train, adj_list_test#,label_list_test

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=1, shuffle=False, mode=None):
        self.dataset = dataset
        self.length = len(dataset)
        if batch_size is None: self.batch_size = len(self.dataset)
        else: self.batch_size = batch_size
        self.index = 0
        self.labels = []
        self.shuffle = shuffle
        self.mode = mode
        self.reset()
        self.num_workers = num_workers
    
    def reset(self):
        if self.shuffle: np.random.shuffle(self.dataset)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.index+1)*self.batch_size > self.length:
            self.reset()
            raise StopIteration
        else:
            batch_data = self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
            if self.mode != "test": self.labels = [graph.label for graph in batch_data]
            self.index += 1
        return self.index-1, self.labels, np.array(batch_data)

def tox21_loader(): #without fingerprint
    adj_list_train = []
    adj_list_test = []
    label_list_train = []
    label_list_test = []
    maxnum_data = 7500
    
    for i in range(maxnum_data):
        try:
            with open("tox21/train/{}_label.txt".format(i), 'r') as f, open("tox21/test/{}_label.txt".format(i), 'r') as f2:
                label_list_train.append(int(f.readline()))
                label_list_test.append(int(f2.readline()))
        except FileNotFoundError:
            try:
                with open("tox21/train/{}_label.txt".format(i), 'r') as f:
                    label_list_train.append(int(f.readline()))
            except FileNotFoundError:
                break
        try:
            nodes_train = []
            nodes_test = []
            with open("tox21/train/{}_graph.txt".format(i), 'r') as f, open("tox21/test/{}_graph.txt".format(i), 'r') as f2:
                dims = [(f,int(f.readline().strip())), (f2,int(f2.readline().strip()))]
                for k, (file,dim) in enumerate(dims):
                    for j in range(dim):
                        row = file.readline().strip().split()
                        #print(row)
                        for item in row:
                            if k: nodes_test.append((j,int(item)))
                            else: nodes_train.append((j,int(item)))
                    if k: adj_list_test.append(Adjacency_List(node_num=dim, adj_list=nodes_test, label=label_list_test[i]))
                    else: adj_list_train.append(Adjacency_List(node_num=dim, adj_list=nodes_train, label=label_list_train[i]))
        except FileNotFoundError: #collect the other train data
            nodes_train = []
            with open("tox21/train/{}_graph.txt".format(i), 'r') as file:
                dim = int(file.readline().strip())
                for j in range(dim):
                    row = file.readline().strip().split()
                    for item in row:
                        nodes_train.append((j,int(item)))
                adj_list_train.append(Adjacency_List(node_num=dim, adj_list=nodes_train, label=label_list_train[i]))

    print("train tox ", np.sum(label_list_train), "in", len(label_list_train))
    print("valid tox ", np.sum(label_list_test), "in", len(label_list_test))
    return adj_list_train, adj_list_test
