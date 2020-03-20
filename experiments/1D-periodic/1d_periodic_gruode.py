import argparse
import gru_ode_bayes.data_utils as data_utils
import gru_ode_bayes
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# parse parameters
parser = argparse.ArgumentParser(description="Running GRUODE on Double OU")
parser.add_argument('--model_name', type=str, help="Model to use", default="double_OU_gru_ode_bayes")
parser.add_argument('--dataset', type=str, help="Dataset CSV file", default="../../gru_ode_bayes/datasets/1D_periodic/1d_periodic.csv")
parser.add_argument('--jitter', type=float, help="Time jitter to add (to split joint observations)", default=0)
parser.add_argument('--seed', type=int, help="Seed for data split generation", default=432)
parser.add_argument('--full_gru_ode', action="store_true", default=True)
parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")
parser.add_argument('--no_impute',action="store_true",default = True)
parser.add_argument('--demo', action = "store_true", default = False)
parser.add_argument('--gpu_num', type=int, default = 0)
parser.add_argument('--sampling_rate', type=float, default = 1)
parser.add_argument('--t', type=int, default = 100)  #timepoint
parser.add_argument('--n', type=int, default = 100) #number of trajectories
parser.add_argument('--niter', type=int, default = 50)

args = parser.parse_args()
if args.demo:
    print(f"Demo Mode - Loading model for double_OU ....")
    gru_ode_bayes.paper_plotting.plot_trained_model(model_name = "double_OU_gru_ode_bayes_demo")
    exit()

# check dataset
# if not default setting, check that the specific dataset exists
if args.sampling_rate!=1 or args.t != 100 or args.n != 100:
    path = "../../gru_ode_bayes/datasets/1D_periodic"
    file_list = os.listdir(path)
    specific_filename="1d_periodic_"+str(args.n)+"_"+str(args.sampling_rate)+".csv"
    if specific_filename in file_list:
        dataset=path+"/"+specific_filename
        print(specific_filename+" is exists, not generating a new dataset")
    else:
        #file generating
        pass

model_name = args.model_name
params_dict=dict()

# device settings
device  = torch.device(f"cuda:{args.gpu_num}")
torch.cuda.set_device(args.gpu_num)


#########################################################################################
#Dataset metadata
#metadata = np.load(f"{args.dataset[:-4]}_metadata.npy",allow_pickle=True).item()

delta_t = 0.05 #metadata["delta_t"]
T       = 5 #metadata["T"]
N=args.n

train_idx, val_idx = train_test_split(np.arange(N),test_size=0.2, random_state=args.seed)   # np.arange(metadata["N"])

# 여기서 val은 validation을 말한다.
val_options = {"T_val": 4, "max_val_samples": 1}

# dataset을 class에 보관하고, validation에 사용하는듯 하다.
data_train = data_utils.ODE_Dataset(csv_file=dataset, idx=train_idx, jitter_time=args.jitter)
# data_val   = data_utils.ODE_Dataset(csv_file=dataset, idx=val_idx, jitter_time=args.jitter, validation = True,
#                                     val_options = val_options )
data_val = data_utils.ODE_Dataset(csv_file=dataset, idx=val_idx, jitter_time=args.jitter)  # validation = False -> for all reconstruction

#Model parameters.
params_dict["input_size"]  = 1  #edited
params_dict["hidden_size"] = 50
params_dict["p_hidden"]    = 25
params_dict["prep_hidden"] = 25
params_dict["logvar"]      = True
params_dict["mixing"]      = 0.0001
params_dict["delta_t"]     = delta_t
params_dict["dataset"]     = args.dataset
params_dict["jitter"]      = args.jitter
#params_dict["gru_bayes"]   = "masked_mlp"
params_dict["full_gru_ode"] = args.full_gru_ode
params_dict["solver"]      = args.solver
params_dict["impute"]      = not args.no_impute

params_dict["T"]           = T
params_dict["samping_rate"] = args.sampling_rate

#Model parameters and the metadata of the dataset used to train the model are stored as a single dictionnary.
# summary_dict ={"model_params":params_dict,"metadata":metadata}
# np.save(f"./../trained_models/{model_name}_params.npy",summary_dict)

dl     = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=500,num_workers=2) #multithread
# data 전체가 batch_size이다.
dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size=len(data_val), num_workers=1)

## the neural negative feedback with observation jumps
model = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                        p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                        logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                        full_gru_ode = params_dict["full_gru_ode"],
                                        solver = params_dict["solver"], impute = params_dict["impute"])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

params_dict=dict()

#Training
for epoch in range(args.niter):
    model.train()
    for i, b in tqdm.tqdm(enumerate(dl)):
        """
        # times = The sorted unique values.
        # time_ptr = 원소들의 누적합
        """
        optimizer.zero_grad()

        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(device)
        M        = b["M"].to(device)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(device)

        y = b["y"]
        hT, loss, _, _  = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov)

        loss.backward()
        optimizer.step()

    #validation result, batch size is 20
    with torch.no_grad():
        mse_val  = 0
        loss_val = 0
        num_obs  = 0
        model.eval()
        for i, b in enumerate(dl_val):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)

            y = b["y"]
            hT, loss, _, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)

            # split p_vec into mean and var
            mean, var = torch.chunk(p_vec, 2, dim=2)    # [time x point]
            # mean, var와 X를 같은 차원으로 만들어준다.
            mean = torch.flatten(mean.squeeze(2))
            var = torch.flatten(var.squeeze(2))
            X = torch.flatten(X)

            mse_loss  = (torch.pow(X - mean, 2)).sum()
            mse_val  += mse_loss.cpu().numpy()
            num_obs  += X.size(0)

        loss_val /= num_obs
        mse_val  /= num_obs
        print(f"Mean validation loss at epoch {epoch}: nll={loss_val:.5f},\
         mse={mse_val:.5f}  (num_interpolated={num_obs}), training_loss={loss:.2f}")

# save validation result
print(f"Last validation log likelihood : {loss_val}")
print(f"Last validation MSE : {mse_val}")
df_file_name = "./../trained_models/1D-periodic.csv"
df_res = pd.DataFrame({"Name" : [model_name], "LogLik" : [loss_val], "MSE" : [mse_val], "Dataset": [args.dataset], "Seed": [args.seed]})
if os.path.isfile(df_file_name):
    df = pd.read_csv(df_file_name)
    df = df.append(df_res)
    df.to_csv(df_file_name,index=False)
else:
    df_res.to_csv(df_file_name,index=False)


model_file = f"./../trained_models/{model_name}.pt"
torch.save(model.state_dict(),model_file)
print(f"Saved model into '{model_file}'.")


"""
Plotting resulting model on newly generated_data
"""
gru_ode_bayes.paper_plotting.plot_trained_model(model_name = model_name)
