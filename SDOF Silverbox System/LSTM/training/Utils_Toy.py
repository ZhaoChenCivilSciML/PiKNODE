
## packages
import torch 
from tqdm import tqdm
import scipy.io
import os

## classes
class MyLSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hidd_layers, OutputWindowSize):
        super(MyLSTM, self).__init__()

        self.input_layer = torch.nn.LSTM(input_size, hidden_size, num_layers = 1 + num_hidd_layers)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)   
        self.OutputWindowSize = OutputWindowSize         
    def forward(self, x):
        h1 = self.input_layer(x)[0] # an LSTM layer outputs a tuple of {output, (hn, cn)}
        
        # only select the last self.OutputWindowSize elements in the sequence of h2 for the many-to-some prediction
        # linear layer takes input of dims: (seq_len, batch, input_size) and gives output of dim (seq_len, batch, output_size)
        y = self.output_layer(h1[-self.OutputWindowSize:, :, :]) 
        
        return y

## functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(mdl_LSTM, x, dx, ddxg, epochs, writer, source_seq, target_seq):
    optimizer = torch.optim.Adam([{'params': mdl_LSTM.parameters()},
                ], lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # normalize data
    x, x_mean, x_std = normalize_data(x)
    dx, dx_mean, dx_std = normalize_data(dx)
    ddxg, ddxg_mean, ddxg_std = normalize_data(ddxg)

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

    # concatenate for LSTM
    X_t_1_seq_tr = torch.cat([x_t_1_seq_tr, dx_t_1_seq_tr], -1)
    X_t_1_seq_val = torch.cat([x_t_1_seq_val, dx_t_1_seq_val], -1)

    for epoch in tqdm(range(epochs)):
        ## training forward pass
        X_t_1_seq_tr_pred = mdl_LSTM(torch.cat([x_t_seq_tr, dx_t_seq_tr, ddxg_t_seq_tr], -1))
        loss_X_tr = loss_fn(X_t_1_seq_tr_pred, X_t_1_seq_tr)

        # total loss
        loss_tr = loss_X_tr

        ## val forward pass
        with torch.no_grad():
            X_t_1_seq_val_pred = mdl_LSTM(torch.cat([x_t_seq_val, dx_t_seq_val, ddxg_t_seq_val], -1))
            loss_X_val = loss_fn(X_t_1_seq_val_pred, X_t_1_seq_val)

        writer.add_scalars('loss_x', {'tr':loss_X_tr.item(), 'val':loss_X_val.item()}, epoch)

        optimizer.zero_grad()
    
        loss_tr.backward()
    
        optimizer.step()

    # evaluate model and report metrics
    with torch.no_grad():
        # sequence forecasting
        seq_err_X_tr = torch.norm(X_t_1_seq_tr_pred.flatten() - X_t_1_seq_tr.flatten())/torch.norm(X_t_1_seq_tr.flatten())*100
        seq_err_X_val = torch.norm(X_t_1_seq_val_pred.flatten() - X_t_1_seq_val.flatten())/torch.norm(X_t_1_seq_val.flatten())*100
        writer.add_text('seq_err_X_tr', 'Train Error(%):' + str(seq_err_X_tr.item()))
        writer.add_text('seq_err_X_val', 'Val Error(%):' + str(seq_err_X_val.item()))

        # open-loop forecasting
        X = torch.cat([x, dx], -1)
        X_t_open = X[:source_seq].unsqueeze(1)
        X_t_1_open = X[source_seq:].unsqueeze(1)
        X_t_1_open_pred_list = []

        ddxg_t_open = ddxg[:source_seq].unsqueeze(1) 
        ddxg_t_1_open = ddxg[source_seq:].unsqueeze(1) 

        for nstep in range(X.shape[0]-source_seq-target_seq+1):
            seq_pred = mdl_LSTM(torch.cat([X_t_open, ddxg_t_open], -1))
            X_t_1_open_pred_list.append(seq_pred)
            X_t_open = torch.cat([X_t_open[1:], seq_pred[[0]]], 0)
            ddxg_t_open = torch.cat([ddxg_t_open[1:], ddxg_t_1_open[[nstep]]], 0)

        X_t_1_open_pred = torch.cat([seq_pred[[0]] for seq_pred in X_t_1_open_pred_list[:-1]] + [X_t_1_open_pred_list[-1]], 0)
        X_open_err = torch.norm(X_t_1_open_pred.flatten() - X_t_1_open.flatten())/torch.norm(X_t_1_open.flatten())*100
        writer.add_text('X_open_err', str(X_open_err.item()))

    # save data
    scipy.io.savemat('pred.mat',{'X_t_1_seq_tr_pred':X_t_1_seq_tr_pred.detach().cpu().numpy(), 'X_t_1_seq_tr':X_t_1_seq_tr.cpu().numpy(),
    'X_t_1_seq_val_pred':X_t_1_seq_val_pred.detach().cpu().numpy(), 'X_t_1_seq_val':X_t_1_seq_val.cpu().numpy(),
    'X_t_1_open_pred': X_t_1_open_pred.detach().cpu().numpy(), 'X_t_1_open': X_t_1_open.cpu().numpy(),
    'x_mean': x_mean.cpu().numpy(), 'x_std': x_std.cpu().numpy(),

    'dx_mean': dx_mean.cpu().numpy(), 'dx_std': dx_std.cpu().numpy(),

    'ddxg_mean': ddxg_mean.cpu().numpy(), 'ddxg_std': ddxg_std.cpu().numpy(),

    })


    # save model
    save_dict = {}
    save_dict['mdl_LSTM'] = mdl_LSTM.state_dict()
    torch.save(save_dict, 'trained.tar')

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

def open_loop_forecast(x, dx, ddxg, source_seq, mdl_LSTM, mdl_hysteresis, dt, x_mean, x_std,
     dx_mean, dx_std, ddxg_mean, ddxg_std, open_loop_horizon):
    if open_loop_horizon == None:
        open_loop_horizon = ddxg.shape[0] - source_seq
    x_t_open, x_t_1_open = batch_data(x, source_seq, open_loop_horizon)
    dx_t_open, dx_t_1_open = batch_data(dx, source_seq, open_loop_horizon)
    ddxg_t_open, ddxg_t_1_open = batch_data(ddxg, source_seq, open_loop_horizon) 
    
    z_t_1_open_pred, x_t_1_open_pred, dx_t_1_open_pred = seq_pred(mdl_LSTM, mdl_hysteresis, x_t_open, dx_t_open, ddxg_t_open, ddxg_t_1_open, open_loop_horizon, dt,
    x_mean, x_std,
    dx_mean, dx_std, ddxg_mean, ddxg_std)

    x_open_err = torch.norm(x_t_1_open_pred.flatten() - x_t_1_open.flatten())/torch.norm(x_t_1_open.flatten())*100
    dx_open_err = torch.norm(dx_t_1_open_pred.flatten() - dx_t_1_open.flatten())/torch.norm(dx_t_1_open.flatten())*100
    return x_t_1_open, x_t_1_open_pred, dx_t_1_open, dx_t_1_open_pred, z_t_1_open_pred, x_open_err, dx_open_err

def finite_diff(v, dt):
    dv = (-v[4:] + 8*v[3:-1] - 8*v[1:-3] + v[:-4])/12/dt # fourth_order_central_diff for [2:-2]
    dv_forward = (-25/12*v[:2] +4*v[1:3] -3*v[2:4] +4/3*v[3:5] -1/4*v[4:6])/dt # fourth order forward diff for [:2]
    dv_backward = (11/6*v[-2:] -3*v[-3:-1] +3/2*v[-4:-2] -1/3*v[-5:-3])/dt # third order backward diff for [-2:]
    dv = torch.cat([dv_forward, dv, dv_backward], 0)
    return dv

def seq_pred(mdl_LSTM, mdl_hysteresis, x_t_seq, dx_t_seq, ddxg_t_seq, ddxg_t_1_seq, target_seq, dt, x_mean, x_std, dx_mean, dx_std, ddxg_mean, ddxg_std):
    z_t_1_seq_pred_list = []
    x_t_1_seq_pred_list = []
    dx_t_1_seq_pred_list = []

    for nstep in range(target_seq):
        z_t = time_delay_hysteresis(x_t_seq, dx_t_seq, ddxg_t_seq, mdl_hysteresis)

        # deep koopman
        x_t = x_t_seq[-1]
        dx_t = dx_t_seq[-1]
        ddxg_t = ddxg_t_seq[-1]
        z_t_1 = mdl_LSTM(z_t, torch.cat([x_t, dx_t, ddxg_t], -1)) # (batches, states)
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
    # predictor
    x_t_1 = x_t + dt*dx_t 

    ddx_t = -z_t + ddxg_t

    dx_t_1 = dx_t + dt*ddx_t

    # corrector
    x_t_1 = x_t + dt/2*(dx_t + dx_t_1)

    ddx_t_1 = -z_t_1 + ddxg_t_1
    dx_t_1 = dx_t + dt/2*(ddx_t + ddx_t_1)

    return x_t_1, dx_t_1

def load_data(file_tag, device, interval = 1):
    data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), file_tag))
    dt = torch.from_numpy(data['dt']).float().to(device)
    ddxg = 101700*torch.from_numpy(data['u'][:, ::interval]).t().float().to(device) # (timesteps, 1)
    x = torch.from_numpy(data['y'][:, ::interval]).t().float().to(device) # (timesteps, DOFs)
    dx = finite_diff(x, dt)
    return x, dx, dt, ddxg

