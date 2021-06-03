from kid_score import calculate_kid_given_data
from fid_score import calculate_fid_given_data
import numpy as np
import torch
from torchvision import datasets

def prepare_real_img(path, data = 'mnist'):
    if data == 'mnist':
        testloader = datasets.MNIST(path, train=False)
    else:
        testloader = datasets.FashionMNIST(path, train=False)
    real_img = []
    for i, j in enumerate(testloader):
        x = np.asarray(j[0])
        real_img.append(x)
    real_img = np.stack(real_img)
    real_img = real_img.reshape(real_img.shape[0],1,28,28)
    real_img = np.pad(real_img,  ((0,0),(0,0),(2,2),(2,2)), 'constant')
    return np.float32(real_img)/ 255.

def prepare_syn_img(path):
    syn_img = np.load(path)['data']
    syn_img = syn_img.reshape(syn_img.shape[0],1,28,28)
    syn_img = np.pad(syn_img,  ((0,0),(0,0),(2,2),(2,2)), 'constant')
    return np.float32(syn_img)

def prepare_real_img_fmnist(path):
    testloader = datasets.FashionMNIST(path, train = False)
    real_img = []
    for i, j in enumerate(testloader):
        x = np.asarray(j[0])
        real_img.append(x)
    real_img = np.stack(real_img)
    real_img = real_img.reshape(real_img.shape[0],1,28,28)
    real_img = np.pad(real_img,  ((0,0),(0,0),(2,2),(2,2)), 'constant')
    real_img = np.repeat(real_img, 3, axis = 1)
    return np.float32(real_img) / 255.
    
def prepare_syn_img_fmnist(path):
    syn_img = np.load(path)['data']
    syn_img = syn_img.reshape(syn_img.shape[0],1,28,28)
    syn_img = np.pad(syn_img,  ((0,0),(0,0),(2,2),(2,2)), 'constant')
    syn_img = np.repeat(syn_img, 3, axis = 1)
    return np.float32(syn_img)

realimg = prepare_real_img('data')
filename = 'synthetic_mnist.npz'
mean, mean2 = [], []
std, std2 = [], []

synimg = prepare_syn_img(filename)
results = calculate_fid_given_data(realimg, synimg, 100, torch.cuda.is_available(), 2048,
                                      model_type = 'lenet')
results2 = calculate_kid_given_data(realimg, synimg, 100, torch.cuda.is_available(), 2048,
                                      model_type = 'lenet')
mean.append(results[0][0])
std.append(results[0][1])
mean2.append(results2[0][0])
std2.append(results2[0][1])   

print(f'the average fid value is {np.mean(mean)} +- {np.sqrt(sum(map(lambda x:x*x, std))/len(std))}')
print(f'the average kid value is {np.mean(mean2)} +- {np.sqrt(sum(map(lambda x:x*x, std2))/len(std2))}')