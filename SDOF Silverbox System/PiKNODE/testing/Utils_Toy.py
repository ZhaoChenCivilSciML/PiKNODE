
## packages
import torch 
from tqdm import tqdm
import scipy.io
import os

## classes
        
class Deep_DMDc(torch.nn.Module):
    # autonomous system
    def __init__(self, N_states, N_control, N_hidd, N_layers, N_modes):
        super(Deep_DMDc, self).__init__()
        # encoders
        self.encoder_list_states = torch.nn.ModuleList([torch.nn.Linear(N_states, N_hidd)] + [torch.nn.Linear(N_hidd, N_hidd) for _ in range(N_layers)]\
             + [torch.nn.Linear(N_hidd, N_modes)]) 

        self.encoder_list_control = torch.nn.ModuleList([torch.nn.Linear(N_control, N_hidd)] + [torch.nn.Linear(N_hidd, N_hidd) for _ in range(N_layers)]\
             + [torch.nn.Linear(N_hidd, N_modes)]) 

        # linear DMD operator for control
        self.DMD_state = torch.nn.Linear(N_modes, N_modes, bias=False) 
        self.DMD_control = torch.nn.Linear(N_modes, N_modes, bias=False) 

        # MLP decoder
        self.decoder_list_states =  torch.nn.ModuleList([torch.nn.Linear(N_modes, N_hidd)] + [torch.nn.Linear(N_hidd, N_hidd) for _ in range(N_layers)]\
             + [torch.nn.Linear(N_hidd, N_states)])

        self.swish = torch.nn.SiLU()
        self.N_modes = N_modes

    def forward(self, X_t, U_t):
        # states X_t: (batch, N_states)
        # control U_t: (batch, N_control)

        # encoder
        H_X_t = self.encoder_forward(X_t, self.encoder_list_states) # (batch, N_modes)
        H_U_t = self.encoder_forward(U_t, self.encoder_list_control) # (batch, N_modes)

        # DMD latent space
        H_X_t_1 = self.Koopman_control(H_X_t, H_U_t)

        # decoder
        X_t_1 = self.decoder_forward(H_X_t_1, self.decoder_list_states) # (batch, N_states)
        return X_t_1

    def Koopman_control(self, H_X_t, H_U_t):
        H_X_t_1 = self.DMD_state(H_X_t) + self.DMD_control(H_U_t) # (batch, N_modes)
        return H_X_t_1

    def encoder_forward(self, X_t, encoder_list):
        # H_t = X_t.permute((1, 0, 2)).reshape((X_t.shape[1], -1))
        H_t = self.MLP_forward(encoder_list, X_t) # (batch, N_modes)
        return H_t

    def MLP_forward(self, list, H_t):
        for layer in list[:-1]:
            H_t = self.swish(layer(H_t))
        H_t = list[-1](H_t) 
        return H_t

    def decoder_forward(self, H_t_1, decoder_list):
        X_t_1 = self.MLP_forward(decoder_list, H_t_1) # (batch, N_inputs)
        return X_t_1

## functions
def train(mdl_DMD, mdl_hysteresis, x, dx, ddxg, epochs, writer, source_seq, target_seq, dt, open_loop_horizon = None, x_mean = None, x_std = None, dx_mean = None,
 dx_std = None, ddxg_mean = None,
 ddxg_std = None):
    optimizer = torch.optim.Adam([{'params': mdl_DMD.parameters()},
                {'params': mdl_hysteresis.parameters()}], lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # normalize data
    x, x_mean, x_std = normalize_data(x, x_mean, x_std)
    dx, dx_mean, dx_std = normalize_data(dx, dx_mean, dx_std)
    ddxg, ddxg_mean, ddxg_std = normalize_data(ddxg, ddxg_mean, ddxg_std)

    # batch data
    x_t_seq, x_t_1_seq = batch_data(x, source_seq, target_seq)
    dx_t_seq, dx_t_1_seq = batch_data(dx, source_seq, target_seq)
    ddxg_t_seq, ddxg_t_1_seq = batch_data(ddxg, source_seq, target_seq)

    # train & val data split: sequential split
    N_batch = x_t_seq.shape[1]
    split_ind = int(0.7*N_batch)

    x_t_seq_tr, x_t_seq_val, x_t_1_seq_tr, x_t_1_seq_val = split_data(x_t_seq, x_t_1_seq, split_ind)
    dx_t_seq_tr, dx_t_seq_val, dx_t_1_seq_tr, dx_t_1_seq_val = split_data(dx_t_seq, dx_t_1_seq, split_ind)
    ddxg_t_seq_tr, ddxg_t_seq_val, ddxg_t_1_seq_tr, ddxg_t_1_seq_val = split_data(ddxg_t_seq, ddxg_t_1_seq, split_ind)

    for epoch in tqdm(range(epochs)):
        ## training forward pass
        loss_x_tr, loss_dx_tr, loss_hys_tr, loss_ae_tr, loss_koopman_tr, z_t_1_seq_tr_pred, x_t_1_seq_tr_pred, dx_t_1_seq_tr_pred = comprehensive_forward_pass(mdl_DMD,
         mdl_hysteresis, x_t_seq_tr, x_t_1_seq_tr, dx_t_seq_tr, dx_t_1_seq_tr, ddxg_t_seq_tr, ddxg_t_1_seq_tr, target_seq, dt, x_mean, x_std,
            dx_mean, dx_std, ddxg_mean, ddxg_std, loss_fn)

        # total loss
        loss_tr = loss_x_tr + loss_dx_tr + loss_hys_tr + loss_ae_tr + loss_koopman_tr

        ## val forward pass
        with torch.no_grad():
            loss_x_val, loss_dx_val, loss_hys_val, loss_ae_val, loss_koopman_val, z_t_1_seq_val_pred, x_t_1_seq_val_pred, dx_t_1_seq_val_pred = comprehensive_forward_pass(mdl_DMD,
            mdl_hysteresis, x_t_seq_val, x_t_1_seq_val, dx_t_seq_val, dx_t_1_seq_val, ddxg_t_seq_val, ddxg_t_1_seq_val, target_seq, dt, x_mean, x_std,
                dx_mean, dx_std, ddxg_mean, ddxg_std, loss_fn)

        writer.add_scalars('loss_x', {'tr':loss_x_tr.item(), 'val':loss_x_val.item()}, epoch)
        writer.add_scalars('loss_dx', {'tr':loss_dx_tr.item(), 'val':loss_dx_val.item()}, epoch)
        writer.add_scalars('loss_ae', {'tr':loss_ae_tr.item(), 'val':loss_ae_val.item()}, epoch)
        writer.add_scalars('loss_hys', {'tr':loss_hys_tr.item(), 'val':loss_hys_val.item()}, epoch)
        writer.add_scalars('loss_koopman', {'tr':loss_koopman_tr.item(), 'val':loss_koopman_val.item()}, epoch)

    # evaluate model and report metrics
    with torch.no_grad():
        # sequence forecasting
        seq_err_x_tr = torch.norm(x_t_1_seq_tr_pred.flatten() - x_t_1_seq_tr.flatten())/torch.norm(x_t_1_seq_tr.flatten())*100
        seq_err_x_val = torch.norm(x_t_1_seq_val_pred.flatten() - x_t_1_seq_val.flatten())/torch.norm(x_t_1_seq_val.flatten())*100
        writer.add_text('seq_err_x_tr', 'Train Error(%):' + str(seq_err_x_tr.item()))
        writer.add_text('seq_err_x_val', 'Val Error(%):' + str(seq_err_x_val.item()))

        seq_err_dx_tr = torch.norm(dx_t_1_seq_tr_pred.flatten() - dx_t_1_seq_tr.flatten())/torch.norm(dx_t_1_seq_tr.flatten())*100
        seq_err_dx_val = torch.norm(dx_t_1_seq_val_pred.flatten() - dx_t_1_seq_val.flatten())/torch.norm(dx_t_1_seq_val.flatten())*100
        writer.add_text('seq_err_dx_tr', 'Train Error(%):' + str(seq_err_dx_tr.item()))
        writer.add_text('seq_err_dx_val', 'Val Error(%):' + str(seq_err_dx_val.item()))

        # open-loop forecasting
        x_t_1_open, x_t_1_open_pred, dx_t_1_open, dx_t_1_open_pred, z_t_1_open_pred, x_open_err, dx_open_err = open_loop_forecast(x, dx, ddxg, source_seq,
         mdl_DMD,
         mdl_hysteresis, dt, x_mean, x_std,
            dx_mean, dx_std, ddxg_mean, ddxg_std, open_loop_horizon)
        writer.add_text('open_loop_err_x', str(x_open_err.item()))
        writer.add_text('open_loop_err_dx', str(dx_open_err.item()))

    # save data
    scipy.io.savemat('pred.mat',{'x_t_1_seq_tr_pred':x_t_1_seq_tr_pred.detach().cpu().numpy(), 'x_t_1_seq_tr':x_t_1_seq_tr.cpu().numpy(),
    'x_t_1_seq_val_pred':x_t_1_seq_val_pred.detach().cpu().numpy(), 'x_t_1_seq_val':x_t_1_seq_val.cpu().numpy(),
    'x_t_1_open_pred': x_t_1_open_pred.detach().cpu().numpy(), 'x_t_1_open': x_t_1_open.cpu().numpy(),
    'x_mean': x_mean.cpu().numpy(), 'x_std': x_std.cpu().numpy(),

    'dx_t_1_seq_tr_pred':dx_t_1_seq_tr_pred.detach().cpu().numpy(), 'dx_t_1_seq_tr':dx_t_1_seq_tr.cpu().numpy(),
    'dx_t_1_seq_val_pred':dx_t_1_seq_val_pred.detach().cpu().numpy(), 'dx_t_1_seq_val':dx_t_1_seq_val.cpu().numpy(),
    'dx_t_1_open_pred': dx_t_1_open_pred.detach().cpu().numpy(), 'dx_t_1_open': dx_t_1_open.cpu().numpy(),
    'dx_mean': dx_mean.cpu().numpy(), 'dx_std': dx_std.cpu().numpy(),

    'ddxg_mean': ddxg_mean.cpu().numpy(), 'ddxg_std': ddxg_std.cpu().numpy(),

    'z_t_1_seq_tr_pred':z_t_1_seq_tr_pred.detach().cpu().numpy(), 
    'z_t_1_seq_val_pred':z_t_1_seq_val_pred.detach().cpu().numpy(), 
    'z_t_1_open_pred': z_t_1_open_pred.detach().cpu().numpy(), 
    })

def split_data(X_t_seq, X_t_1_seq, split_ind):
    X_t_seq_tr = X_t_seq[:, :split_ind, :]
    X_t_seq_val = X_t_seq[:, split_ind:, :]

    X_t_1_seq_tr = X_t_1_seq[:, :split_ind, :]
    X_t_1_seq_val = X_t_1_seq[:, split_ind:, :]
    return X_t_seq_tr, X_t_seq_val, X_t_1_seq_tr, X_t_1_seq_val

def batch_data(ah, source_seq, target_seq):
    x_t_seq = torch.stack([ah[i:i+source_seq] for i in range(ah.shape[0]-source_seq-target_seq+1)], dim=1) # (sequence, batch, N_inputs)
    x_t_1_seq = torch.stack([ah[i+source_seq:i+source_seq+target_seq] for i in range(ah.shape[0]-source_seq-target_seq+1)], dim=1)
    return x_t_seq, x_t_1_seq

def normalize_data(ah, x_mean = None, x_std = None):
    if x_mean == None and x_std == None:
        x_mean = torch.mean(ah, dim=0, keepdim=True)
        x_std = torch.std(ah, dim=0, keepdim=True)
    ah = (ah - x_mean)/x_std # (timesteps, state)
    return ah, x_mean, x_std

def de_normalize_data(ah, x_mean, x_std):
    return ah*x_std + x_mean

def open_loop_forecast(x, dx, ddxg, source_seq, mdl_DMD, mdl_hysteresis, dt, x_mean, x_std,
     dx_mean, dx_std, ddxg_mean, ddxg_std, open_loop_horizon):
    if open_loop_horizon == None:
        open_loop_horizon = ddxg.shape[0] - source_seq
    x_t_open, x_t_1_open = batch_data(x, source_seq, open_loop_horizon)
    dx_t_open, dx_t_1_open = batch_data(dx, source_seq, open_loop_horizon)
    ddxg_t_open, ddxg_t_1_open = batch_data(ddxg, source_seq, open_loop_horizon) 
    
    z_t_1_open_pred, x_t_1_open_pred, dx_t_1_open_pred = seq_pred(mdl_DMD, mdl_hysteresis, x_t_open, dx_t_open, ddxg_t_open, ddxg_t_1_open, open_loop_horizon, dt,
    x_mean, x_std,
    dx_mean, dx_std, ddxg_mean, ddxg_std)

    x_open_err = torch.norm(x_t_1_open_pred.flatten() - x_t_1_open.flatten())/torch.norm(x_t_1_open.flatten())*100
    dx_open_err = torch.norm(dx_t_1_open_pred.flatten() - dx_t_1_open.flatten())/torch.norm(dx_t_1_open.flatten())*100
    return x_t_1_open, x_t_1_open_pred, dx_t_1_open, dx_t_1_open_pred, z_t_1_open_pred, x_open_err, dx_open_err

def comprehensive_forward_pass(mdl_DMD, mdl_hysteresis, x_t_seq, x_t_1_seq, dx_t_seq, dx_t_1_seq, ddxg_t_seq, ddxg_t_1_seq, target_seq, dt, x_mean, x_std,
     dx_mean, dx_std, ddxg_mean, ddxg_std, loss_fn):
    # target sequence prediction 
    z_t_1_seq_pred, x_t_1_seq_pred, dx_t_1_seq_pred = seq_pred(mdl_DMD, mdl_hysteresis, x_t_seq, dx_t_seq, ddxg_t_seq, ddxg_t_1_seq, target_seq, dt, x_mean, x_std,
     dx_mean, dx_std, ddxg_mean, ddxg_std)

    loss_x = loss_fn(x_t_1_seq_pred, x_t_1_seq)
    loss_dx = loss_fn(dx_t_1_seq_pred, dx_t_1_seq)

    # hysteresis constraint
    z_t = time_delay_hysteresis(x_t_seq, dx_t_seq, ddxg_t_seq, mdl_hysteresis) # (batches, states)
    dx_t = dx_t_seq[-1]
    ddx_t = finite_diff(de_normalize_data(dx_t, dx_mean, dx_std), dt)
    EOM_hys = ddx_t + de_normalize_data(z_t, ddxg_mean, ddxg_std) - de_normalize_data(ddxg_t_seq[-1], ddxg_mean, ddxg_std)

    loss_hys = loss_fn(normalize_data(EOM_hys, ddxg_mean, ddxg_std)[0], torch.zeros_like(EOM_hys))
    
    # autoencoder loss
    H_X_t = mdl_DMD.encoder_forward(z_t, mdl_DMD.encoder_list_states)
    z_t_ae = mdl_DMD.decoder_forward(H_X_t, mdl_DMD.decoder_list_states)
    loss_ae = loss_fn(z_t, z_t_ae)

    # koopman loss
    x_t = x_t_seq[-1]
    ddxg_t = ddxg_t_seq[-1]
    U_t = torch.cat([x_t, dx_t, ddxg_t], -1)
    H_U_t = mdl_DMD.encoder_forward(U_t, mdl_DMD.encoder_list_control) 
    H_X_t_1 = mdl_DMD.Koopman_control(H_X_t, H_U_t)
    loss_koopman = loss_fn(H_X_t[1:], H_X_t_1[:-1])

    return loss_x, loss_dx, loss_hys, loss_ae, loss_koopman, z_t_1_seq_pred, x_t_1_seq_pred, dx_t_1_seq_pred

def finite_diff(v, dt):
    dv = (-v[4:] + 8*v[3:-1] - 8*v[1:-3] + v[:-4])/12/dt # fourth_order_central_diff for [2:-2]
    dv_forward = (-25/12*v[:2] +4*v[1:3] -3*v[2:4] +4/3*v[3:5] -1/4*v[4:6])/dt # fourth order forward diff for [:2]
    dv_backward = (11/6*v[-2:] -3*v[-3:-1] +3/2*v[-4:-2] -1/3*v[-5:-3])/dt # third order backward diff for [-2:]
    dv = torch.cat([dv_forward, dv, dv_backward], 0)
    return dv

def seq_pred(mdl_DMD, mdl_hysteresis, x_t_seq, dx_t_seq, ddxg_t_seq, ddxg_t_1_seq, target_seq, dt, x_mean, x_std, dx_mean, dx_std, ddxg_mean, ddxg_std):
    z_t_1_seq_pred_list = []
    x_t_1_seq_pred_list = []
    dx_t_1_seq_pred_list = []

    for nstep in range(target_seq):
        z_t = time_delay_hysteresis(x_t_seq, dx_t_seq, ddxg_t_seq, mdl_hysteresis)

        # deep koopman
        x_t = x_t_seq[-1]
        dx_t = dx_t_seq[-1]
        ddxg_t = ddxg_t_seq[-1]
        z_t_1 = mdl_DMD(z_t, torch.cat([x_t, dx_t, ddxg_t], -1)) # (batches, states)
        z_t_1_seq_pred_list.append(z_t_1)

        # timestepper of EOMs
        ddxg_t_1 = ddxg_t_1_seq[nstep]
        x_t_1, dx_t_1 = EOM_timestepper(de_normalize_data(x_t, x_mean, x_std), de_normalize_data(dx_t, dx_mean, dx_std), de_normalize_data(z_t, ddxg_mean, ddxg_std),
         de_normalize_data(z_t_1, ddxg_mean, ddxg_std), de_normalize_data(ddxg_t, ddxg_mean, ddxg_std), de_normalize_data(ddxg_t_1, ddxg_mean, ddxg_std), dt)

        x_t_1, _, _ = normalize_data(x_t_1, x_mean, x_std)
        dx_t_1, _, _ = normalize_data(dx_t_1, dx_mean, dx_std)

        x_t_1_seq_pred_list.append(x_t_1) # next step prediction. (batch, states)
        dx_t_1_seq_pred_list.append(dx_t_1) # next step prediction. (batch, states)

        # skip the first timestep and concatenate with a new one
        x_t_seq = torch.cat([x_t_seq[1:], x_t_1.unsqueeze(0)], dim=0)
        dx_t_seq = torch.cat([dx_t_seq[1:], dx_t_1.unsqueeze(0)], dim=0)
        ddxg_t_seq = torch.cat([ddxg_t_seq[1:], ddxg_t_1_seq[[nstep]]],dim=0)

    z_t_1_seq_pred = torch.stack(z_t_1_seq_pred_list, dim=0)
    x_t_1_seq_pred = torch.stack(x_t_1_seq_pred_list, dim=0)
    dx_t_1_seq_pred = torch.stack(dx_t_1_seq_pred_list, dim=0)
    return z_t_1_seq_pred, x_t_1_seq_pred, dx_t_1_seq_pred # (seq, batches, states)

def time_delay_hysteresis(x_t_seq, dx_t_seq, ddxg_t_seq, mdl_hysteresis):
    # time-delay transformation for latent hysteresis variable
    x_t_td = x_t_seq.permute((1, 0, 2)).reshape((x_t_seq.shape[1], -1)) # td: time-delay
    dx_t_td = dx_t_seq.permute((1, 0, 2)).reshape((dx_t_seq.shape[1], -1)) # (seq, batch, states) -> (batches, seq*states)
    ddxg_t_td = ddxg_t_seq.permute((1, 0, 2)).reshape((ddxg_t_seq.shape[1], -1))
    z_t = mdl_hysteresis(torch.cat([x_t_td, dx_t_td, ddxg_t_td], -1)) # (batches, states)
    return z_t 

def EOM_timestepper(x_t, dx_t, z_t, z_t_1, ddxg_t, ddxg_t_1, dt):
    # (batches, states)
    ddx_t = -z_t + ddxg_t
    ddx_t_1 = -z_t_1 + ddxg_t_1
    dx_t_1 = dx_t + dt/2*(ddx_t + ddx_t_1)
    x_t_1 = x_t + dt/2*(dx_t + dx_t_1)
    return x_t_1, dx_t_1

def load_data(file_tag, device, interval = 1):
    data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), file_tag))
    dt = torch.from_numpy(data['dt']).float().to(device)
    ddxg = 101700*torch.from_numpy(data['u'][:, ::interval]).t().float().to(device) # (timesteps, 1)
    x = torch.from_numpy(data['y'][:, ::interval]).t().float().to(device) # (timesteps, DOFs)
    dx = finite_diff(x, dt)
    return x, dx, dt, ddxg

