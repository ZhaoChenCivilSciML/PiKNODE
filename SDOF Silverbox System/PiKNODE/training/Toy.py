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
file_tag = 'Schroeder80mV_long.mat'
x, dx, dt, ddxg = load_data(file_tag, device)

## add noise
noise_lvl = 0
x = x + torch.std(x, 0)*noise_lvl*torch.randn_like(x)
dx = dx + torch.std(dx, 0)*noise_lvl*torch.randn_like(dx)
ddxg = ddxg + torch.std(ddxg, 0)*noise_lvl*torch.randn_like(ddxg)

## set up Deep DMDc models
N_states, N_control = dx.shape[1], dx.shape[1] + x.shape[1] + ddxg.shape[1]
N_hidd, N_layers, N_modes = 32, 3, 32
source_seq, target_seq = 128, 64
mdl_DMD = Deep_DMDc(N_states, N_control, N_hidd, N_layers, N_modes).to(device)

## time-delay NN
mdl_hysteresis = torch.nn.Sequential(torch.nn.Linear(N_control*source_seq, N_hidd),
    torch.nn.SiLU(),
    torch.nn.Linear(N_hidd, N_hidd),
    torch.nn.SiLU(),
    torch.nn.Linear(N_hidd, N_hidd),
    torch.nn.SiLU(),
    torch.nn.Linear(N_hidd, N_states),
     ).to(device)

#%% count trainable paras
no_para_DMD = count_parameters(mdl_DMD)
no_para_hys = count_parameters(mdl_hysteresis)
no_para_total = no_para_DMD + no_para_hys


#%%
## train
start_time = time.time()

epochs = 10000 # 10000
open_loop_horizon = None
train(mdl_DMD, mdl_hysteresis, x, dx, ddxg, epochs, writer, source_seq, target_seq, dt, open_loop_horizon)

## timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()


#%%