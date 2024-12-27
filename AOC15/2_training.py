import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
import tqdm
import model as model

epochs = 25
batch_size = 64
lr = 1.2e-3
wd = 1e-2
warmup_epochs = 5

r = 0.6
seq_len = 256
hw_rate = 10**-1

need_test = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate data
def get_dataloader(csv, is_train=False):
    eps = 1e-5
    df = pd.read_csv(csv, encoding='utf-8')
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 2, int(r * 1392))
    Y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    # generate X_min, X_max, Y_min, Y_max
    if is_train:
        global X_min, X_max, Y_min, Y_max
        X_min = torch.min(X, dim=0, keepdim=True).values
        X_max = torch.max(X, dim=0, keepdim=True).values
        Y_min = torch.min(Y, dim=0, keepdim=True).values
        Y_max = torch.max(Y, dim=0, keepdim=True).values
    # normalization
    X = (X - X_min) / (X_max - X_min + eps)
    Y = (Y - Y_min) / (Y_max - Y_min + eps)
    # data augmentation
    noise_level = 1e-2
    if is_train:
        X = torch.cat((X, torch.randn_like(X) * noise_level + X), dim=0)
        Y = torch.cat((Y, torch.randn_like(Y) * noise_level + Y), dim=0)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return dataloader

train_dataloader = get_dataloader('./data/train_data', True)
valid_dataloader = get_dataloader('./data/valid_data')
test_dataloader = get_dataloader('./data/test_data')

# net, optimizer and criterion
net = model.CNN(seq_len, hw_rate).to(device)
param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Model Param Num: ', param_num)
criterion = model.MyCustomLoss(Y_min, Y_max).to(device)
optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# one epoch
def one_epoch(data_loader, is_train=False):
    tmp_rmse_loss_list = list()
    tmp_mae_loss_list = list()
    tmp_mape_loss_list = list()
    for data in data_loader:
        X, Y = data[0].to(device), data[1].to(device)
        if is_train: optimizer.zero_grad()
        output = net(X)
        rmse_loss, mae_loss, mape_loss = criterion(output, Y)
        back_loss = rmse_loss
        if is_train: back_loss.backward()
        if is_train: optimizer.step()
        with torch.no_grad():
            tmp_rmse_loss_list.append(rmse_loss.item())
            tmp_mae_loss_list.append(mae_loss.item())
            tmp_mape_loss_list.append(mape_loss.item())
    return np.mean(tmp_rmse_loss_list), np.mean(tmp_mae_loss_list), np.mean(tmp_mape_loss_list)

# train & valid
print('\nTraining...')
for i in tqdm.tqdm(range(epochs)):
    # train
    net.train()
    train_rmse_loss, train_mae_loss, train_mape_loss = one_epoch(train_dataloader, True)
    scheduler.step()
    # validate
    net.eval()
    with torch.no_grad():
        valid_rmse_loss, valid_mae_loss, valid_mape_loss = one_epoch(valid_dataloader, False)
    print('Train RMSE Loss: ', train_rmse_loss * 4.1893)
    print('Train MAE  Loss: ', train_mae_loss * 4.1893)
    print('Train MAPE Loss: ', train_mape_loss * 100)
    print('Valid RMSE Loss: ', valid_rmse_loss * 4.1893)
    print('Valid MAE  Loss: ', valid_mae_loss * 4.1893)
    print('Valid MAPE Loss: ', valid_mape_loss * 100)
print('End Training')

# test
if need_test:
    print('\nTesting...')
    net.eval()
    with torch.no_grad():
        test_rmse_loss, test_mae_loss, test_mape_loss = one_epoch(test_dataloader, False)
    print('Test  RMSE Loss: ', test_rmse_loss * 4.1893)
    print('Test  MAE  Loss: ', test_mae_loss * 4.1893)
    print('Test  MAPE Loss: ', test_mape_loss * 100)
    print('End Testing')