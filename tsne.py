# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from cProfile import label
from time import time
from turtle import shape
from importlib_metadata import files
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn import datasets
from sklearn.manifold import TSNE
import os
 
def get_data():
    dir = os.path.join(os.getcwd(),'MMA','MMA_test.txt')
    
    with open(dir,'r') as f :
        files = f.readlines()
    data = np.empty((0,1024*6))
    label = []
    for file in files :
        file = file.replace('\n', '')
        motion = file.strip().split('_')[0]
        file = '.'.join([file,'txt'])
        
        file_path = os.path.join('MMA',motion,file)
        
        points = np.loadtxt(file_path,delimiter=',')
        points= np.reshape(points,(1,6144))
        data = np.vstack((data,points))
        label.append(mylabel(motion))
    label = np.array(label)
    np.save('data.npy',data)
    np.save('label.npy',label)
    

def mylabel(motion):
    dic = {'boxing':0,'jack':1,'jump':2,'squats':3,'walk':4}
    return dic[motion]

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
 
 
def main():
    
    data = np.load('data.npy')
   
    label = np.load('label.npy')
    
    

 
 
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()
 
 
if __name__ == '__main__':
    main()