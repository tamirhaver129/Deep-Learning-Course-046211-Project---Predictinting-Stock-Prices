import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
import math, time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import optuna


seed = 211
np.random.seed(seed)
torch.manual_seed(seed)


def load_nasdaq(use_fed_data=True, use_dif = True):
    filepath = 'input/NASDAQ_DAYLY.csv'
    nasdaq = pd.read_csv(filepath)
    nasdaq['Date'] = pd.to_datetime(nasdaq['Date'], format='%Y-%m-%d')
    # nasdaq = nasdaq[nasdaq['Date'] > datetime.datetime(2018, 1, 1)]
    nasdaq = nasdaq.set_index('Date')
    nasdaq = nasdaq[['Close']]
    filepath = 'input/WALCL.csv'
    assets = pd.read_csv(filepath)
    assets['DATE'] = pd.to_datetime(assets['DATE'], format='%Y-%m-%d')
    assets = assets.rename(columns={'DATE': 'Date', "WALCL": 'Millions_of_Dollars'})
    assets.set_index('Date', inplace=True)
    nasdaq['Assets'] = assets['Millions_of_Dollars']
    nasdaq = nasdaq.dropna()
    if use_dif:
        nasdaq['Diff'] = nasdaq.diff()['Close']
        nasdaq = nasdaq.dropna()
    scaled = nasdaq.copy()
    scaler1 = MinMaxScaler(feature_range=(min(nasdaq['Close']) / max(nasdaq['Close']), 1))
    # scaler1 = MinMaxScaler(feature_range=(0.1, 1))
    scaler2 = MinMaxScaler(feature_range=(min(nasdaq['Assets']) / max(nasdaq['Assets']), 1))
    # scaler2 = MinMaxScaler(feature_range=(0.1, 1))
    scaled['Close'] = scaler1.fit_transform(scaled['Close'].values.reshape(-1, 1))
    scaled['Assets'] = scaler2.fit_transform(scaled['Assets'].values.reshape(-1, 1))
    if use_dif:
        scaled['Diff'] = scaler1.transform(scaled['Diff'].values.reshape(-1, 1))
    if not use_fed_data:
        del nasdaq['Assets']
        del scaled['Assets']


    return nasdaq, scaled, scaler1, scaler2


def load_data(stock, look_back, device, batch_size=1, prev_perm=None, use_perm=True):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back + 1])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - 2 * (test_set_size)
    np.random.seed(seed)
    if prev_perm is not None:
        rand_perm = prev_perm
    else:
        rand_perm = np.random.permutation(data.shape[0])
    if use_perm:
        data = data[rand_perm, :, :]

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)
    x_val = data[train_set_size:train_set_size + test_set_size, :-1, :]
    y_val = data[train_set_size:train_set_size + test_set_size, -1, 0].reshape(-1, 1)
    x_test = data[train_set_size + test_set_size:, :-1]
    y_test = data[train_set_size + test_set_size:, -1, 0].reshape(-1, 1)

    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)
    x_val = torch.from_numpy(x_val).type(torch.Tensor).to(device)
    y_val = torch.from_numpy(y_val).type(torch.Tensor).to(device)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=batch_size)

    return x_test, y_test, x_train, y_train, x_val, y_val, train_loader

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        torch.manual_seed(seed)
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        torch.manual_seed(seed)
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def pct_deviation(prediction,target):
    return np.mean(np.abs(prediction-target)/target)

def train_iter(train_loader,model,optimizer,x_val,y_val):
    """
            for batch_idx, (data,target) in enumerate(train_loader):

                # Forward pass
                y_train_pred = model(data)
                MSE_loss = loss_fn(y_train_pred, target)
                loss = torch.sqrt(MSE_loss)
                train_loss += MSE_loss.item()
                # Zero out gradient, else they will accumulate between epochs
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

            train_loss = math.sqrt(train_loss/len(train_loader))
            model.eval()
            y_val_pred = model(x_val)
            val_loss = torch.sqrt(loss_fn(y_val_pred, y_val)).item()
            """
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        y_train_pred = model(data)
        loss = torch.mean(torch.abs(y_train_pred - target) / target)
        train_loss += loss.item()
        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    model.eval()
    y_val_pred = model(x_val)
    val_loss = torch.mean(torch.abs(y_val_pred - y_val) / y_val)
    return val_loss, train_loss

def objective(trial, model_name, device, input_dim,scaled):
    look_back = trial.suggest_int("Look Back", 10, 60, step=5)
    batch_size = trial.suggest_int('Batch Size', 16, 128)
    hidden_dim = trial.suggest_int('Hidden Dim', 32, 128)
    num_layers = trial.suggest_int("N layers", 1, 3)
    lr = trial.suggest_float("Learning Rate", 1e-4, 0.1, log=True)
    num_iters = 3000
    factor = trial.suggest_float('Factor', 0.55, 0.95, step=0.1)

    x_test, y_test, x_train, y_train, x_val, y_val, train_loader = load_data(scaled, look_back, device, batch_size,
                                                                             use_perm=True)
    output_dim = 1
    if model_name == "GRU":
        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                    num_layers=num_layers, device=device).to(device)
    elif model_name == "LSTM":
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                     num_layers=num_layers, device=device).to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = min(num_iters // len(train_loader), 100)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=5)

    for epoch in range(num_epochs):
        model.train()
        val_loss, train_loss = train_iter(train_loader,model,optimizer,x_val,y_val)
        scheduler.step(val_loss)
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss

def train(best_trial, model_name, device, input_dim,scaled,scaler1):
    look_back = best_trial.params['Look Back']
    batch_size = best_trial.params['Batch Size']
    hidden_dim = best_trial.params['Hidden Dim']
    num_layers = best_trial.params['N layers']
    lr = best_trial.params['Learning Rate']
    num_iters = 3000
    factor = best_trial.params['Factor']
    # dropout = best_trial.params['Dropout']

    x_test, y_test, x_train, y_train, x_val, y_val, train_loader = load_data(scaled, look_back, device, batch_size,
                                                                             use_perm=True)
    output_dim = 1
    if model_name == "GRU":
        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                    num_layers=num_layers, device=device).to(device)
    elif model_name == "LSTM":
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                     num_layers=num_layers, device=device).to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = min(num_iters // len(train_loader), 100)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=5)
    hist = np.zeros(num_epochs)
    print("\nTraining Loss:")
    for epoch in range(num_epochs):
        model.train()
        val_loss, train_loss = train_iter(train_loader,model,optimizer,x_val,y_val)
        scheduler.step(val_loss)

        hist[epoch] = val_loss
        if epoch == num_epochs - 1:
            print("Last Epoch", epoch, "Loss: ", train_loss)

    # plot loss
    plt.plot(hist, label="Training loss")
    plt.legend()
    plt.show()

    # make predictions
    model.eval()
    y_test_pred = model(x_test)
    y_train_pred = model(x_train)
    y_val_pred = model(x_val)
    # invert predictions

    y_train_pred = scaler1.inverse_transform(y_train_pred.cpu().detach().numpy())
    y_train = scaler1.inverse_transform(y_train.detach().cpu().numpy())
    y_test_pred = scaler1.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test = scaler1.inverse_transform(y_test.cpu().detach().numpy())
    y_val_pred = scaler1.inverse_transform(y_val_pred.cpu().detach().numpy())
    y_val = scaler1.inverse_transform(y_val.cpu().detach().numpy())
    return y_train_pred, y_train, y_test_pred, y_test, y_val_pred, y_val, model

def plot_results(model, look_back, nasdaq, scaled, scaler1, device, last = 20):
    x_test, y_test, x_train, y_train, x_val, y_val, train_loader= load_data(scaled, look_back, device, use_perm = False)
    x = torch.cat((x_train,x_val,x_test))
    y = torch.cat((y_train,y_val,y_test))
    model.eval()
    y_pred = model(x)
    y_pred = scaler1.inverse_transform(y_pred.cpu().detach().numpy())
    y = scaler1.inverse_transform(y.cpu().detach().numpy())
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(nasdaq[-len(y):].index, y, color = 'red', label = 'Real NASDAQ Price')
    axes.plot(nasdaq[-len(y):].index, y_pred, color = 'blue', label = 'Predicted NASDAQ Price')
    #axes.xticks(np.arange(0,394,50))
    plt.title('NASDAQ-100 Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('NASDAQ-100 Stock Price')
    plt.legend()
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(nasdaq[-last:].index, y[-last:], '.', color = 'red', label = 'Real NASDAQ Price')
    axes.plot(nasdaq[-last:].index, y_pred[-last:], '.', color = 'blue', label = 'Predicted NASDAQ Price')
    plt.title('NASDAQ-100 Stock Price Prediction - Last Weeks')
    plt.xlabel('Time')
    plt.ylabel('NASDAQ-100 Stock Price')
    plt.legend()

def predict_results(model, look_back, scaled, scaler1, use_fed_data, use_dif, device, n=5):
    data_raw = scaled.values
    predictions = []
    data = np.array(data_raw[len(data_raw) - look_back:])
    x_test = torch.from_numpy(data).type(torch.Tensor).to(device)
    x_test = torch.unsqueeze(x_test, dim=0)
    for _ in range(n):
        y_test_pred = model(x_test)
        cur_pred = y_test_pred[0, 0].item()
        tmp = [cur_pred]
        if use_fed_data:
            tmp.append(2 * x_test[0, -1, 1].item() - x_test[0, -2, 1].item())
        if use_dif:
            tmp.append(cur_pred-x_test[0,-1,0].item())

        tmp = torch.unsqueeze(torch.unsqueeze(torch.tensor(tmp),0),0).to(device)
        x_test = torch.cat((x_test, tmp), 1)[:, 1:, :]
        y_test_pred = scaler1.inverse_transform(y_test_pred.cpu().detach().numpy())
        predictions.append(y_test_pred.item())

    figure, axes = plt.subplots(figsize=(15, 6))
    axes.plot(range(1, n + 1), predictions)
    plt.xticks(range(1, n + 1))
    plt.title('NASDAQ-100 Stock Price Future Prediction - Last Weeks')
    plt.xlabel('Weeks Ahead')
    plt.ylabel('NASDAQ-100 Stock Price')
    plt.show()
    print(predictions)


def study_results(study, model_name, use_fed_data, use_dif, device ,n=5, last=20, plot=True):
    nasdaq, scaled, scaler1, _ = load_nasdaq(use_fed_data, use_dif)
    input_dim = 1 +use_dif +use_fed_data
    print("Best trial:")
    best_trial = study.best_trial
    look_back = best_trial.params['Look Back']
    print("Validation Loss: ", best_trial.value)
    print("Best trial Parameters:")
    print(best_trial.params)
    y_train_pred, y_train, y_test_pred, y_test, y_val_pred, y_val, model = train(best_trial, model_name,device,input_dim,scaled,scaler1)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    trainPctDev = pct_deviation(y_train_pred[:,0],y_train[:,0])
    print('\n\nModel Train Score: %.2f RMSE' % (trainScore))
    print('Model Train Percentage Points deviation: %.2f%%\n' % (100*trainPctDev))
    valScore = math.sqrt(mean_squared_error(y_val[:,0], y_val_pred[:,0]))
    valPctDev = pct_deviation(y_val_pred[:,0],y_val[:,0])
    print('Model Validation Score: %.2f RMSE' % (valScore))
    print('Model Validation Percentage Points deviation: %.2f%%\n' % (100*valPctDev))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    testPctDev = pct_deviation(y_test_pred[:,0],y_test[:,0])
    print('Model Test Score: %.2f RMSE' % (testScore))
    print('Model Test Percentage Points deviation: %.2f%%\n' % (100*testPctDev))
    y = np.concatenate((y_train,y_val,y_test))
    y_pred = np.concatenate((y_train_pred,y_val_pred,y_test_pred))
    totScore = math.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))
    totPctDev = pct_deviation(y_pred,y)
    print('Aggregated Score: %.2f RMSE' % (totScore))
    print('Aggregated Percentage Points deviation: %.2f%%\n\n' % (100*totPctDev))
    if plot:
        plot_results(model, look_back, nasdaq, scaled, scaler1, device, last=last)
        predict_results(model, look_back, scaled, scaler1, use_fed_data, use_dif, device, n=n)