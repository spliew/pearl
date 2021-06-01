### this is for training a single generator for all labels ###
""" with the analysis of """
### weights = weights + N(0, sigma**2*(sqrt(2)/N)**2)
### columns of mean embedding = raw + N(0, sigma**2*(2/N)**2)

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import socket
from sdgym import load_dataset
import argparse
import sys, os
import pickle


import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from field import CategoricalField, NumericalField

from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator
from autodp.mechanism_zoo import ComposedGaussianMechanism

import warnings
warnings.filterwarnings('ignore')

import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# specify heterogeneous dataset or not
heterogeneous_datasets = [ 'adult']
homogeneous_datasets = []

def reverse_data(sample, fields):
    current_ind = 0
    sample_data = []
    for i in range(len(fields)): 
        dim = fields[i].dim()
        data_totranse = sample[:,current_ind:(current_ind+dim)]
        sample_data.append(pd.DataFrame(fields[i].reverse(data_totranse)))
        current_ind = current_ind+ dim
    sample_data = pd.concat(sample_data,axis=1)
    return sample_data.to_numpy()

def normalize(data, num_num, nummode = 'minmax', **kwargs):
    fields = []
    feed_data = []
    data = pd.DataFrame(data)
    for i in range(data.shape[1]):
        if i < num_num:
            col2 = NumericalField(model=nummode)
            col2.get_data(data[i])
            fields.append(col2)
            col2.learn()
            feed_data.append(col2.convert(np.asarray(data[i])))
        else:
            col1 = CategoricalField("one-hot", noise=None)
            fields.append(col1)
            col1.get_data(data[i])
            col1.learn()
            features = col1.convert(np.asarray(data[i]))
            cols = features.shape[1]
            rows = features.shape[0]
            for j in range(cols):
                feed_data.append(features.T[j])
    feed_data = pd.DataFrame(feed_data).T
    return feed_data.to_numpy(), fields

def normalize_only(data, fields):
    data = pd.DataFrame(data)
    outputs = []
    for i in range(len(fields)):
        if fields[i].dtype == "Categorical Data":
            features = fields[i].convert(np.asarray(data[i]))
            cols = features.shape[1]
            for j in range(cols):
                outputs.append(features.T[j])
        else:
            outputs.append(fields[i].convert(np.asarray(data[i])))


    feed_data = pd.DataFrame(outputs).T
    return feed_data.to_numpy()

def save_mydataset(a,b,c,d, name, path="/home1/irteamsu/work/dp-merf/code_tab/tab_data/recorded/"):
    with open(path+name+"_xtrain.p","wb") as f:
        pickle.dump(a,f)
    with open(path+name+"_ytrain.p","wb") as f:
        pickle.dump(c,f)
    with open(path+name+"_xtest.p","wb") as f:
        pickle.dump(b,f)
    with open(path+name+"_ytest.p","wb") as f:
        pickle.dump(d,f)
    print(f'saved files to {path}')

def load_mydataset(name, path="/home1/irteamsu/work/dp-merf/code_tab/tab_data/recorded/"):
    with open(path+name+"_xtrain.p","rb") as f:
        a = pickle.load(f)
    with open(path+name+"_ytrain.p","rb") as f:
        c = pickle.load(f)
    with open(path+name+"_xtest.p","rb") as f:
        b = pickle.load(f)
    with open(path+name+"_ytest.p","rb") as f:
        d = pickle.load(f)
    return a, b, c, d


args=argparse.ArgumentParser()

args.add_argument("--dataset", default="adult")
args.add_argument("--private", type=int, default=1)
args.add_argument("--epochs", default='8000')
args.add_argument("--batch", type=float, default=0.1)
args.add_argument("--num_features", default='1000')
args.add_argument("--undersample", type=float, default=0.4)


args.add_argument("--syn_path", type=str, default="logs/exp_fullrf_adversarial/")
args.add_argument("--repeat", type=int, default=5)
#args.add_argument('--classifiers', nargs='+', type=int, help='list of integers', default=[2])
args.add_argument('--classifiers', nargs='+', type=int, help='list of integers', default=[0])
args.add_argument("--data_type", default='generated') #both, real, generated
arguments=args.parse_args()
print("arg", arguments)
list_epochs=[int(i) for i in arguments.epochs.split(',')]
list_num_features=[int(i) for i in arguments.num_features.split(',')]

############################### kernels to use ###############################
""" we use the random fourier feature representation for Gaussian kernel """

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

    W = torch.Tensor(W).to(device)
    X = X.to(device)
    XWT = torch.mm(X, torch.t(W)).to(device)
    
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
    return Z


""" we use a weighted polynomial kernel for labels """

def Feature_labels(labels, weights):

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    labels = labels.to(device)

    weighted_labels_feature = labels/weights

    return weighted_labels_feature


############################### end of kernels ###############################

############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size,
                fields):
        super(Generative_Model_homogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

        self.fields = fields

        # self.dataset = dataset


    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        # if self.dataset=='credit':
        #     all_pos = self.relu(output[:,-1])
        #     output = torch.cat((output[:,:-1], all_pos[:,None]),1)
        outputs = []
        current_ind = 0
        for i in range(len(self.fields)):
            if self.fields[i].dtype == "Categorical Data":
                dim = self.fields[i].dim()
                outputs.append(torch.softmax(output[:,current_ind:current_ind+dim],dim=1))
                current_ind = current_ind+dim    
            else: 
                outputs.append(torch.tanh(output[:,current_ind:current_ind+1]))
                current_ind = current_ind +1
        output_combined = torch.cat(outputs,dim=1)
        # output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
        # output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
        # output_combined = torch.cat((output_numerical, output_categorical), 1)

        return output_combined




class Generative_Model_heterogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, 
                    num_categorical_inputs, num_numerical_inputs,
                    fields,
                    ):
        super(Generative_Model_heterogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.num_numerical_inputs = num_numerical_inputs
        self.num_categorical_inputs = num_categorical_inputs
        self.fields = fields

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        # self.sigmoid = torch.nn.Sigmoid()



    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)
        outputs = []
        current_ind = 0
        for i in range(len(self.fields)):
            if self.fields[i].dtype == "Categorical Data":
                dim = self.fields[i].dim()
                outputs.append(torch.softmax(output[:,current_ind:current_ind+dim],dim=1))
                current_ind = current_ind+dim    
            else: 
                outputs.append(torch.tanh(output[:,current_ind:current_ind+1]))
                current_ind = current_ind +1
        output_combined = torch.cat(outputs,dim=1)
        # output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
        # output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
        # output_combined = torch.cat((output_numerical, output_categorical), 1)

        return output_combined

############################### end of generative models ###############################



def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected






######################################## beginning of main script ############################################################################

def main(dataset, undersampled_rate, n_features_arg, mini_batch_size_arg, how_many_epochs_arg, is_priv_arg, seed_number):

    random.seed(seed_number)

    is_private = is_priv_arg

    ############################### data loading ##################################


    if dataset=='adult':

        data, categorical_columns, ordinal_columns = load_dataset('adult')
        # else:

        """ some specifics on this dataset """
        categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
        n_classes = 2


        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        X_train, X_test, y_train, y_test = load_mydataset(dataset)

        X_train, fields = normalize(X_train, num_numerical_inputs)


    ##########################################################################################3



    def test_models(X_tr, y_tr, X_te, y_te, datasettype):

        print("\n",  datasettype, "data\n")

        roc_arr=[]
        prc_arr=[]
        f1_arr=[]

        aa=np.array([1,3])

        models=np.array([GradientBoostingClassifier(), MLPClassifier()])
        models_to_test=models[np.array(arguments.classifiers)]

        # check
        for model in models_to_test:
        #for model in [LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02)]:

            print('\n', type(model))
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)  # test on real data


            if n_classes>2:

                f1score = f1_score(y_te, pred, average='weighted')

                print("F1-score on test %s data is %.3f" % (datasettype, f1score))
                f1_arr.append(f1score)

            else:

                roc = roc_auc_score(y_te, pred)
                prc = average_precision_score(y_te, pred)

                print("ROC on test %s data is %.3f" % (datasettype, roc))
                print("PRC on test %s data is %.3f" % (datasettype, prc))

                roc_arr.append(roc)
                prc_arr.append(prc)

        if n_classes > 2:

            res1 = np.mean(f1_arr); res1_arr=f1_arr
            print("------\nf1 mean across methods is %.3f\n" % res1)
            res2 = 0; res2_arr=0  # dummy
        else:

            res1=np.mean(roc_arr); res1_arr=roc_arr
            res2=np.mean(prc_arr); res2_arr=prc_arr
            print("-"*40)
            print("roc mean across methods is %.3f" % res1)
            print("prc mean across methods is %.3f\n" % res2)

        return res1_arr, res2_arr
        #return res1, res2


    #######################################################
    # TEST REAL DATA


    # check
    if arguments.data_type=='real' or arguments.data_type=='both':
        if dataset in heterogeneous_datasets:
            roc_real, prc_real = test_models(X_train, y_train, X_test, y_test, "real")
            roc_return, prc_return=roc_real, prc_real
        else:
            f1_real = test_models(X_train, y_train, X_test, y_test, "real")
            f1_return = f1_real



    ###########################################################################
    # PREPARING GENERATOR

    if arguments.data_type=='generated' or arguments.data_type=='both':

        # one-hot encoding of labels.
        n, input_dim = X_train.shape
        print(f'xtrain shape {X_train.shape}')
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = np.expand_dims(y_train, 1)
        true_labels = onehot_encoder.fit_transform(y_train)


        ######################################
        # MODEL

        # model specifics
        mini_batch_size = np.int(np.round(mini_batch_size_arg * n))
        print("minibatch: ", mini_batch_size)
        input_size = 10 + 1
        hidden_size_1 = 4 * input_dim
        hidden_size_2 = 2 * input_dim
        output_size = input_dim

        if dataset in homogeneous_datasets:

            model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                          hidden_size_2=hidden_size_2,
                                                          output_size=output_size, fields = fields).to(device)

        elif dataset in heterogeneous_datasets:

            model = Generative_Model_heterogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                            hidden_size_2=hidden_size_2,
                                                            output_size=output_size,
                                                            num_categorical_inputs=num_categorical_inputs,
                                                            num_numerical_inputs=num_numerical_inputs,
                                                            fields = fields).to(device)
        else:
            print('sorry, please enter the name of your dataset either in homogeneous_dataset or heterogeneous_dataset list ')

        # define details for training
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        how_many_epochs = how_many_epochs_arg
        how_many_iter = 1 #np.int(n / mini_batch_size)
        training_loss_per_epoch = np.zeros(how_many_epochs)


        ##########################################################################


        """ specifying random fourier features """

        idx_rp = np.random.permutation(n)

        if dataset=='census': # some columns of census data have many zeros, so we need more datapoints to get meaningful length scales
            num_data_pt_to_discard = 100
        else:
            num_data_pt_to_discard = 10

        idx_to_discard = idx_rp[0:num_data_pt_to_discard]
        idx_to_keep = idx_rp[num_data_pt_to_discard:]

        if dataset !='adult':
            print('error')
            sys.exit()
        else:

            if dataset in heterogeneous_datasets:
                med = util.meddistance(X_train[idx_to_discard, ])
            else:
                med = util.meddistance(X_train[idx_to_discard, ])

            sigma2 = med ** 2

        X_train = X_train[idx_to_keep,:]
        true_labels = true_labels[idx_to_keep,:]
        n = X_train.shape[0]
        print('total number of datapoints in the training data is', n)

        # random Fourier features
        n_features = n_features_arg
        draws = n_features // 2

        # random fourier features for numerical inputs only
        if dataset in heterogeneous_datasets:
            W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)
        else:
            W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

        """ specifying ratios of data to generate depending on the class lables """
        unnormalized_weights = np.sum(true_labels,0)
        weights = unnormalized_weights/np.sum(unnormalized_weights)
        print('\nweights with no privatization are', weights, '\n'  )

        ####################################################
        # Privatising quantities if necessary

        """ privatizing weights """
        if is_private:
            print("private")
            # desired privacy level
            epsilon = 1.0
            delta = 1e-5
            # k = n_classes + 1   # this dp analysis has been updated
            k = 2
            # privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
            # print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

            general_calibrate = generalized_eps_delta_calibrator()
            params = {}
            params['sigma'] = None
            params['coeff'] = k

            mech3 = general_calibrate(ComposedGaussianMechanism, epsilon, delta, [0,1000],params=params,para_name='sigma', name='Composed_Gaussian')
            print(mech3.name, mech3.params, mech3.get_approxDP(delta))
            privacy_param = mech3.params

            sensitivity_for_weights = np.sqrt(2)/n  # double check if this is sqrt(2) or 2
            noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
            weights = weights + np.random.randn(weights.shape[0])*noise_std_for_weights
            weights[weights < 0] = 1e-3 # post-processing so that we don't have negative weights.
            print('weights after privatization are', weights)

        """ computing mean embedding of subsampled true data """
        if dataset in homogeneous_datasets:

            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(X_train), W_freq)
            emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
            outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
            mean_emb1 = torch.mean(outer_emb1, 0)

        else:  # heterogeneous data

            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(X_train), W_freq)

            # numerical_input_data = X_train[:, 0:num_numerical_inputs]

            # emb1_numerical = (RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)).to(device)

            # categorical_input_data = X_train[:, num_numerical_inputs:]

            # emb1_categorical = (torch.Tensor(categorical_input_data) / np.sqrt(num_categorical_inputs)).to(device)

            # emb1_input_features = torch.cat((emb1_numerical, emb1_categorical), 1)

            emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
            outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
            mean_emb1 = torch.mean(outer_emb1, 0)


        """ privatizing each column of mean embedding """
        if is_private:
            # if dataset in heterogeneous_datasets:
            #     sensitivity = 2*np.sqrt(2) / n
            # else:
            #     sensitivity = 2 / n
            # homogeneous
            sensitivity = 2 / n
            noise_std_for_privacy = privacy_param['sigma'] * sensitivity

            # make sure add noise after rescaling
            weights_torch = torch.Tensor(weights)
            weights_torch = weights_torch.to(device)

            rescaled_mean_emb = weights_torch*mean_emb1
            noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
            noise = noise.to(device)

            rescaled_mean_emb = rescaled_mean_emb + noise

            mean_emb1 = rescaled_mean_emb/weights_torch # rescaling back\

        # End of Privatising quantities if necessary
        ####################################################

        ##################################################################################################################
        # TRAINING THE GENERATOR

        print('Starting Training')
        inner_iter = 10
        optsig = torch.ones(input_dim).to('cuda:0') * sigma2
        optsig.requires_grad = True
        optimizer2 = torch.optim.Adam([optsig], lr=0.01)

        orisig = torch.ones(input_dim).to('cuda:0') * sigma2
        orisig.requires_grad = False

        ww = torch.from_numpy(W_freq  ** 2 /2).type(torch.FloatTensor).to(device)
        ww.requires_grad = False
        alpha = torch.nn.Softmax(dim=0)(ww @ (orisig - optsig))
        alpha = alpha.unsqueeze(1).repeat(1,2)

        def get_alpha(x, w = ww, sigma = sigma2):
            orisig = torch.ones(x.shape[0]).to(x.device) * sigma
            orisig.requires_grad = False
            alpha = torch.nn.Softmax(dim=0)(w @ (orisig - x))
            alpha = alpha.unsqueeze(1).repeat(1,2)
            return alpha
        n_cate = draws

        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                """ computing mean embedding of generated data """
                # zero the parameter gradients
                model.train()
                optsig.requires_grad = False
                optimizer.zero_grad()

                if dataset in homogeneous_datasets: # In our case, if a dataset is homogeneous, then it is a binary dataset.
                    
                    label_input = (1 * (torch.rand((mini_batch_size)) < weights[1])).type(torch.FloatTensor)
                    label_input = label_input.to(device)
                    feature_input = torch.randn((mini_batch_size, input_size-1)).to(device)
                    input_to_model = torch.cat((feature_input, label_input[:,None]), 1)

                    #we feed noise + label (like in cond-gan) as input
                    outputs = model(input_to_model)

                    """ computing mean embedding of generated samples """
                    emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

                    label_input_t = torch.zeros((mini_batch_size, n_classes))
                    idx_1 = (label_input == 1.).nonzero()[:,0]
                    idx_0 = (label_input == 0.).nonzero()[:,0]
                    label_input_t[idx_1, 1] = 1.
                    label_input_t[idx_0, 0] = 1.

                    emb2_labels = Feature_labels(label_input_t, weights)
                    outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                    mean_emb2 = torch.mean(outer_emb2, 0)

                else:  # heterogeneous data

                    # (1) generate labels
                    label_input = torch.multinomial(torch.Tensor([weights]), mini_batch_size, replacement=True).type(torch.FloatTensor)
                    label_input=torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),1) #to avoid no labels
                    label_input = label_input.transpose_(0,1)
                    label_input = label_input.to(device)

                    # (2) generate corresponding features
                    feature_input = torch.randn((mini_batch_size+len(weights), input_size-1)).to(device)
                    input_to_model = torch.cat((feature_input, label_input), 1)
                    # emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                    outputs = model(input_to_model)
                    # (3) compute the embeddings of those
                    # numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
                    # emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq) #W_freq [n_features/2,6], n_features=10000

                    # categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]

                    # emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

                    # emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)
                    emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

                    generated_labels = onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
                    emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
                    outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                    mean_emb2 = torch.mean(outer_emb2, 0)
                # loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2
                loss = (mean_emb1 - mean_emb2)  ** 2
                loss = loss[n_cate:,:] + loss[:n_cate, :]
                alpha = get_alpha(optsig)
                loss = alpha * ( loss ) * n_cate
                loss = torch.sum(loss) 

                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                torch.cuda.empty_cache()

            if epoch % inner_iter  == 0:
                model.eval()
                optsig.requires_grad = True
                optimizer2.zero_grad()
        
        
                label_input = torch.multinomial(torch.Tensor([weights]), mini_batch_size, replacement=True).type(torch.FloatTensor)
                label_input=torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),1) #to avoid no labels
                label_input = label_input.transpose_(0,1)
                label_input = label_input.to(device)

                # (2) generate corresponding features
                feature_input = torch.randn((mini_batch_size+len(weights), input_size-1)).to(device)
                input_to_model = torch.cat((feature_input, label_input), 1)
                # emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                outputs = model(input_to_model)
                # (3) compute the embeddings of those
                # numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
                # emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq) #W_freq [n_features/2,6], n_features=10000

                # categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]

                # emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

                # emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)
                emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

                generated_labels = onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
                emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
                outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                mean_emb2 = torch.mean(outer_emb2, 0)
                
                
                
                loss = (mean_emb1 - mean_emb2)  ** 2
                loss = loss[n_cate:,:] + loss[:n_cate, :]
                alpha = get_alpha(optsig)
                loss = -alpha * ( loss ) * n_cate
                loss = torch.sum(loss) 
                loss.backward()
                optimizer2.step()


                torch.cuda.empty_cache()

            if epoch % 100 == 0:
                print('epoch # and running loss are ', [epoch, running_loss])
                training_loss_per_epoch[epoch] = running_loss



        #######################################################################33
        if dataset in heterogeneous_datasets:

            """ draw final data samples """

            label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
            label_input = label_input.transpose_(0, 1)
            label_input = label_input.to(device)

            # (2) generate corresponding features
            feature_input = torch.randn((n, input_size - 1)).to(device)
            input_to_model = torch.cat((feature_input, label_input), 1)
            outputs = model(input_to_model)


            # (3) round the categorial features
            # output_numerical = outputs[:, 0:num_numerical_inputs]
            # output_categorical = outputs[:, num_numerical_inputs:]
            # output_categorical = torch.round(output_categorical)

            # output_combined = torch.cat((output_numerical, output_categorical), 1)

            # generated_input_features_final = output_combined.cpu().detach().numpy()
            generated_input_features_final = outputs.cpu().detach().numpy()
            X_final = reverse_data(generated_input_features_final, fields)

            generated_labels_final = label_input.cpu().detach().numpy()
            # save data
            savex = "{}_{}_{}_{}_{}_{}_{}_xsyntrain.p".format(dataset, undersampled_rate, n_features_arg, 
                                                            mini_batch_size_arg, how_many_epochs_arg,
                                                             is_priv_arg,seed_number )
            savey = "{}_{}_{}_{}_{}_{}_{}_ysyntrain.p".format(dataset, undersampled_rate, n_features_arg,
                                                             mini_batch_size_arg, how_many_epochs_arg, 
                                                             is_priv_arg, seed_number)
            # with open(os.path.join(arguments.syn_path,dataset+"_xsyntrain.p"), "wb") as f:
            with open(os.path.join(arguments.syn_path,savex), "wb") as f:
                pickle.dump(X_final, f)
            # with open(os.path.join(arguments.syn_path,dataset+"_ysyntrain.p"), "wb") as f:
            with open(os.path.join(arguments.syn_path,savey), "wb") as f:
                pickle.dump(generated_labels_final, f)

            #### make data one hot etc before passing it to classifier
            X_test = normalize_only(X_test, fields)


            roc, prc= test_models(generated_input_features_final, generated_labels_final, X_test, y_test, "generated")
            roc_return, prc_return=roc, prc
            #roc, prc=-1, -1

            # LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
            # LR_model_ours.fit(generated_input_features_final, generated_labels_final)  # training on synthetic data
            # pred_ours = LR_model_ours.predict(X_test)  # test on real data

            #return roc_real, prc_real
            #return roc, prc

        else: # homogeneous datasets

            """ now generate samples from the trained network """

            # weights[1] represents the fraction of the positive labels in the dataset,
            #and we would like to generate a similar fraction of the postive/negative datapoints
            label_input = (1 * (torch.rand((n)) < weights[1])).type(torch.FloatTensor)
            label_input = label_input.to(device)

            feature_input = torch.randn((n, input_size - 1)).to(device)
            input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
            outputs = model(input_to_model)

            samp_input_features = outputs

            label_input_t = torch.zeros((n, n_classes))
            idx_1 = (label_input == 1.).nonzero()[:, 0]
            idx_0 = (label_input == 0.).nonzero()[:, 0]
            label_input_t[idx_1, 1] = 1.
            label_input_t[idx_0, 0] = 1.

            samp_labels = label_input_t

            generated_input_features_final = samp_input_features.cpu().detach().numpy()
            generated_input_features_final = reverse_data(generated_input_features_final, fields)

            generated_labels_final = samp_labels.cpu().detach().numpy()
            generated_labels=np.argmax(generated_labels_final, axis=1)
            # save data
            savex = "{}_{}_{}_{}_{}_{}_{}_xsyntrain.p".format(dataset, undersampled_rate, n_features_arg, 
                                                            mini_batch_size_arg, how_many_epochs_arg,
                                                             is_priv_arg,seed_number )
            savey = "{}_{}_{}_{}_{}_{}_{}_ysyntrain.p".format(dataset, undersampled_rate, n_features_arg,
                                                             mini_batch_size_arg, how_many_epochs_arg, 
                                                             is_priv_arg, seed_number)
            # with open(os.path.join(arguments.syn_path,dataset+"_xsyntrain.p"), "wb") as f:
            with open(os.path.join(arguments.syn_path,savex), "wb") as f:
                pickle.dump(generated_input_features_final, f)
            # with open(os.path.join(arguments.syn_path,dataset+"_ysyntrain.p"), "wb") as f:
            with open(os.path.join(arguments.syn_path,savey), "wb") as f:
                pickle.dump(generated_labels_final, f)
            # with open(os.path.join(arguments.syn_path,dataset+"_xsyntrain.p"), "wb") as f:
            #     pickle.dump(generated_input_features_final, f)
            # with open(os.path.join(arguments.syn_path,dataset+"_ysyntrain.p"), "wb") as f:
            #     pickle.dump(generated_labels, f)

            f1 = test_models(generated_input_features_final, generated_labels, X_test, y_test, "generated")
            f1_return=f1
            #f1=-1

            #return f1_real
            #return f1

    # returnt he values, if dataset is real, then real f1/roc/prc are returned, if dataset is generated or both geenrated f1/roc/prc are returned

    if dataset in heterogeneous_datasets:
        return roc_return, prc_return
    else:
        return f1_return


        #LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
        #LR_model_ours.fit(generated_input_features_final,
         #                 np.argmax(generated_labels_final, axis=1))  # training on synthetic data
        #pred_ours = LR_model_ours.predict(X_test)  # test on real data



    # if n_classes>2:
    #
    #     f1score = f1_score(y_test, pred_ours, average='weighted')
    #     print('F1-score (ours) is ', f1score)
    #
    #     return f1score
    #
    # else:
    #
    #     roc = roc_auc_score(y_test, pred_ours)
    #     prc = average_precision_score(y_test, pred_ours)
    #
    #     print('ROC ours is', roc)
    #     print('PRC ours is', prc)
    #     print('n_features are ', n_features)
    #
    #     return roc, prc


###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == '__main__':



    is_priv_arg = arguments.private  # check
    single_run = True# check

    repetitions = arguments.repeat  # check

    # check
    #for dataset in ["credit", "epileptic", "census", "cervical", "adult", "isolet", "covtype", "intrusion"]:
    for dataset in [arguments.dataset]:
    #for dataset in ["intrusion"]:
        print("\n\n")
        print('is private?', is_priv_arg)

        if single_run == True:
            how_many_epochs_arg = list_epochs#[200]
            # n_features_arg = [100000]#, 5000, 10000, 50000, 80000]
            n_features_arg = list_num_features#[500]
            mini_batch_arg = [arguments.batch]#[0.3]
            undersampling_rates = [arguments.undersample] #[.8]  # dummy
        else:
            how_many_epochs_arg = [8000, 6000, 2000, 1000, 4000]
            n_features_arg = [500, 1000, 2000, 5000, 10000, 50000, 80000, 100000]
            # n_features_arg = [5000, 10000, 50000, 80000, 100000]
            # n_features_arg = [50000, 80000, 100000]
            mini_batch_arg = [0.5]

            if dataset=='adult':
                mini_batch_arg=[0.1]
                n_features_arg = [500, 1000, 2000, 5000, 10000, 50000]
                undersampling_rates = [.3]#[.8, .6, .4] #dummy



        grid = ParameterGrid({"undersampling_rates": undersampling_rates, "n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
                              "how_many_epochs_arg": how_many_epochs_arg})





        if dataset in ["adult"]:

            max_aver_roc, max_aver_prc, max_roc, max_prc, max_aver_rocprc, max_elem=0, 0, 0, 0, [0,0], 0

            for elem in grid:
                print(elem, "\n")
                prc_arr_all = []; roc_arr_all = []
                for ii in range(repetitions):
                    print("\nRepetition: ",ii)

                    roc, prc  = main(dataset, elem["undersampling_rates"], elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)

                    # print("Variables: ")
                    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                    #                          key=lambda x: -x[1])[:10]:
                    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                    # print("\n")

                    roc_arr_all.append(roc)
                    prc_arr_all.append(prc)


                roc_each_method_avr=np.mean(roc_arr_all, 0)
                prc_each_method_avr=np.mean(prc_arr_all, 0)
                roc_each_method_std = np.std(roc_arr_all, 0)
                prc_each_method_std = np.std(prc_arr_all, 0)
                roc_arr=np.mean(roc_arr_all, 1)
                prc_arr = np.mean(prc_arr_all, 1)

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average ROC each method: ", roc_each_method_avr);
                print("Average PRC each method: ", prc_each_method_avr);
                print("Std ROC each method: ", roc_each_method_std);
                print("Std PRC each method: ", prc_each_method_std)


                print("Average over repetitions across all methods")
                print("Average ROC: ", np.mean(roc_arr)); print("Average PRC: ", np.mean(prc_arr))
                print("Std ROC: ", np.std(roc_arr)); print("Variance PRC: ", np.std(prc_arr), "\n")
                print("\n", "-" * 80, "\n\n\n")

                if np.mean(roc_arr)>max_aver_roc:
                    max_aver_roc=np.mean(roc_arr)

                if np.mean(prc_arr)>max_aver_prc:
                    max_aver_prc=np.mean(prc_arr)

                if np.mean(roc_arr) + np.mean(prc_arr)> max_aver_rocprc[0]+max_aver_rocprc[1]:
                    max_aver_rocprc = [np.mean(roc_arr), np.mean(prc_arr)]
                    max_elem = elem

            print("\n\n", "*"*30, )
            print(dataset)
            print("Max ROC! ", max_aver_rocprc[0])
            print("Max PRC! ", max_aver_rocprc[1])
            print("Setup: ", max_elem)
            print('*'*100)





        else: # multi-class classification problems.

            max_f1, max_aver_f1, max_elem=0, 0, 0

            for elem in grid:
                print(elem, "\n")
                f1score_arr_all = []
                for ii in range(repetitions):
                    print("\nRepetition: ",ii)

                    f1scr = main(dataset, elem["undersampling_rates"], elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)
                    f1score_arr_all.append(f1scr[0])

                f1score_each_method_avr = np.mean(f1score_arr_all, 0)
                f1score_each_method_std = np.std(f1score_arr_all, 0)
                f1score_arr = np.mean(f1score_arr_all, 1)

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average F1: ", f1score_each_method_avr)
                print("Std F1: ", f1score_each_method_std)


                print("Average over repetitions across all methods")
                print("Average f1 score: ", np.mean(f1score_arr))
                print("Std F1: ", np.std(f1score_arr))
                print("\n","-" * 80, "\n\n\n")

                if np.mean(f1score_arr)>max_aver_f1:
                    max_aver_f1=np.mean(f1score_arr)
                    max_elem = elem

            print("\n\n", "Max F1! ", max_aver_f1, "*"*20)
            print("Setup: ", max_elem)
            print('*' * 30)




