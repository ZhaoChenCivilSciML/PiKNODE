#%% packages
import scipy.io
import torch 
import os
from Utils_Toy import *
from torch.utils.tensorboard import SummaryWriter
import time

## fix random seeds
torch.manual_seed(0)

## log training statistics
writer = SummaryWriter("DeepDMD")

## select device
if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")

## load data
file_tag = 'SNLS80mV_truncated.mat'
x, dx, dt, ddxg = load_data(file_tag, device)
# dt = torch.mean(t[1:]-t[:-1])

## add noise
noise_lvl = 0
x = x + torch.std(x, 0)*noise_lvl*torch.randn_like(x)
dx = dx + torch.std(dx, 0)*noise_lvl*torch.randn_like(dx)
ddxg = ddxg + torch.std(ddxg, 0)*noise_lvl*torch.randn_like(ddxg)

## set up Deep DMDc models
N_states, N_control = dx.shape[1], dx.shape[1] + x.shape[1] + ddxg.shape[1]
N_hidd, N_layers, N_modes = 32, 3, 32
source_seq, target_seq = 128, 64

## set up LSTM
mdl_LSTM = MyLSTM(input_size = N_control, hidden_size = N_hidd, output_size = x.shape[1]+dx.shape[1], 
               num_hidd_layers = N_layers, OutputWindowSize = target_seq).to(device)

## load trained models
checkpoint = torch.load('trained.tar', map_location=device)
mdl_LSTM.load_state_dict(checkpoint['mdl_LSTM']) 

## load normalization scales
data_tr = scipy.io.loadmat('pred_train.mat') 
x_mean = torch.from_numpy(data_tr['x_mean']).float().to(device)
x_std = torch.from_numpy(data_tr['x_std']).float().to(device)
dx_mean = torch.from_numpy(data_tr['dx_mean']).float().to(device)
dx_std = torch.from_numpy(data_tr['dx_std']).float().to(device)
ddxg_mean = torch.from_numpy(data_tr['ddxg_mean']).float().to(device)
ddxg_std = torch.from_numpy(data_tr['ddxg_std']).float().to(device)

## train
start_time = time.time()

epochs = 1 # 10000
train(mdl_LSTM, x, dx, ddxg, epochs, writer, source_seq, target_seq, x_mean, x_std, dx_mean, dx_std, ddxg_mean, ddxg_std)

## timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()


#%%