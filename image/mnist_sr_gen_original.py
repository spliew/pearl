import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from models_gen import ConvCondGen
from aux import plot_mnist_batch, meddistance, log_args, flat_data, log_final_score
from data_loading import get_mnist_dataloaders
from rff_mmd_approx import get_rff_losses
from synth_data_benchmark import test_gen_data


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data):
  n_iter = n_data // batch_size
  for batch_idx in range(n_iter):
    gen_code, gen_labels = gen.get_code(batch_size, device)
    loss = rff_mmd_loss(gen(gen_code), gen_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device):
  bs = labels.shape[0]
  gen_code, gen_labels = gen.get_code(bs, device)
  gen_samples = gen(gen_code)
  return rff_mmd_loss(data, labels, gen_samples, gen_labels)


def train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval):

  for batch_idx, (data, labels) in enumerate(train_loader):
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      n_data = len(train_loader.dataset)
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, log_dir):
  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data, labels = data.to(device), labels.to(device)
      data = flat_data(data, labels, device, n_labels=10, add_label=False)
      loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
      test_loss += loss.item()  # sum up batch loss

  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data.cpu().numpy()
#   med_dist = meddistance(data_enc_batch)
#   print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
  gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}', denorm=False)
  print('Test set: Average loss: {:.4f}'.format(test_loss))
  return test_loss



def preprocess_args(ar):
  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)

  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)
  assert ar.data in {'digits', 'fashion'}
  if ar.rff_sigma is None:
    ar.rff_sigma = '105' if ar.data == 'digits' else '127'


def synthesize_mnist_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  assert n_data % gen_batch_size == 0
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def main():
  # load settings
  n_data, n_feat = 60000, 784

  ar = get_args()
  pt.manual_seed(ar.seed)
  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  # load data
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda, dataset=ar.data,
                                                    flip=ar.flip_mnist)

  # init model
  gen = ConvCondGen(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes).to(device)

  # define loss function

  sr_loss, mb_loss, _ = get_rff_losses(train_loader, n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor,
                                       ar.mmd_type)

  # rff_mmd_loss = get_rff_mmd_loss(n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  # training loop
  for epoch in range(1, ar.epochs + 1):
    train_single_release(gen, device, optimizer, epoch, sr_loss, ar.log_interval, ar.batch_size, n_data)
    test(gen, device, test_loader, mb_loss, epoch, ar.batch_size, ar.log_dir)
    scheduler.step()

  # save trained model and data
  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')

  syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, device)
  np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)

  test_model_key = 'mlp' if ar.flip_mnist else 'logistic_reg'
  final_score = test_gen_data(ar.log_name, ar.data, subsample=0.1, custom_keys=test_model_key, data_from_torch=True)
  log_final_score(ar.log_dir, final_score)

