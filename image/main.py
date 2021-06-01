import numpy as np
import torch
import torch.nn.functional as F
from models_gen import ConvCondGen, FCCondGen
from aux import plot_mnist_batch, meddistance, log_args, flat_data, log_final_score
from data_loading import get_mnist_dataloaders, get_2d_synth_dataloaders
from rff_mmd_approx import get_rff_losses, data_label_embedding
from synth_data_benchmark import test_gen_data

from mnist_sr_gen_original import test, synthesize_mnist_with_uniform_labels
from aux import log_final_score

from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd


batch_size= 100
test_batch_size= 1000

rff_param_tuple = namedtuple('rff_params', ['w', 'b'])

seed = 0
filepath = 'result/mnist'
torch.manual_seed(seed)
np.random.seed(seed)
train_loader, test_loader = get_mnist_dataloaders(batch_size, test_batch_size,True, dataset= 'fashion',
                                                   data_dir = '/data', flip= False)

gen = ConvCondGen(5, '200', 10, '16,8', '5,5').to('cuda:0')

  # init optimizer
optimizer = torch.optim.Adam(list(gen.parameters()), lr=0.01)

sr_loss, mb_loss, noisy_emb, wfreq = get_rff_losses(train_loader, 784, 3000,'129', torch.device('cuda:0'), 10, 25.3, # 0.75, # 25.3, # 5.72,
                                       'sphere')

optsig = torch.ones(784).to('cuda:0') *129
optsig.requires_grad = True
optimizer2 = torch.optim.Adam([optsig], lr=0.01)


orisig = torch.ones(784).to('cuda:0') * 129
orisig.requires_grad = False
ww = wfreq.w ** 2 /2
ww.requires_grad = False
alpha = torch.nn.Softmax(dim=0)(ww @ (orisig - optsig))
alpha = alpha.unsqueeze(1).repeat(1,10)

def get_alpha(x, w = ww, sigma = 129 ):
    orisig = torch.ones(x.shape[0]).to(x.device) * sigma
    orisig.requires_grad = False
    alpha = torch.nn.Softmax(dim=0)(w @ (orisig - x))
    alpha = alpha.unsqueeze(1).repeat(1,10)
    return alpha

n_cate = noisy_emb.shape[0] //2

for i in range(5):
        
    for j in range(600):
        # gen opt loop
        gen.train()
        optsig.requires_grad = False
        gen_code, gen_labels = gen.get_code(batch_size, 'cuda:0')
        gen_emb = data_label_embedding(gen(gen_code), gen_labels, wfreq, 'sphere',labels_to_one_hot=False,
                                        n_labels=10, device='cuda:0' )
        loss =   ( gen_emb - noisy_emb )**2
        loss = loss[n_cate:,:] + loss[:n_cate, :]
        alpha = get_alpha(optsig)
        loss = alpha * ( loss ) * 500
        losses = torch.sum(loss)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if j % 5 == 0:
            # dist opt loop
            gen.eval()
            optsig.requires_grad = True
            
            gen_code, gen_labels = gen.get_code(batch_size, 'cuda:0')
            gen_emb = data_label_embedding(gen(gen_code), gen_labels, wfreq, 'sphere',labels_to_one_hot=False,
                                        n_labels=10, device='cuda:0' )
            loss =   ( gen_emb - noisy_emb )**2
            loss = loss[n_cate:,:] + loss[:n_cate, :]
            alpha = get_alpha(optsig)
            loss = - alpha * ( loss ) * 500
            losses = torch.sum(loss)
            optimizer2.zero_grad()
            losses.backward()
            optimizer2.step()
        
    scheduler.step()
    print(f'loss at epoch {i} is {torch.sum( ( gen_emb - noisy_emb ) **2)}')
#     print(f'largest loss at epoch {i} is {torch.max( ( gen_emb - noisy_emb ) **2)} at {torch.argmax( ( gen_emb - noisy_emb ) **2)}')
    test(gen, 'cuda:0', test_loader, mb_loss, i, 1000, filepath+f'/{seed}/')
torch.save(gen.state_dict(), filepath+f'/{seed}/model.pth')
print('saved model!')

syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, 'cuda:0')
np.savez(filepath+f'/{seed}/synthetic_mnist', data=syn_data, labels=syn_labels)