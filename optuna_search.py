import utils
import numpy as np
import random
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import optuna
import joblib
from utils import load_nasdaq, load_data, GRU, LSTM, pct_deviation, objective, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def conduct_study (study_name, model_name,input_dim,scaled, n_trials = 100, timeout=180*60):
    sampler = optuna.samplers.TPESampler(seed=utils.seed)
    study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler)
    print(study_name)
    study.optimize(lambda trial: objective(trial,model_name,device,input_dim,scaled), n_trials, timeout=timeout)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    return study

def hyperparameter_tuning():
    n_trials = 1000
    res_dir = 'studies_try/'
    use_dif = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ### GRU hyperparameter search
    nasdaq, scaled, scaler1, scaler2 = load_nasdaq(use_dif)
    input_dim = 2+use_dif
    model_name = 'GRU'
    study_name = 'study_gru'
    study_gru = conduct_study(study_name, model_name,input_dim,scaled, n_trials)
    end = '_pctloss.pkl'
    joblib.dump(study_gru, os.path.join(res_dir, study_name + end))

    ### LSTM hyperparameter search
    model_name = 'LSTM'
    study_name = 'study_lstm'
    study_lstm = conduct_study(study_name, model_name,input_dim,scaled, n_trials)
    joblib.dump(study_lstm, os.path.join(res_dir, study_name + end))

    ### GRU without Fed hyperparameter search
    nasdaq, scaled, scaler1, scaler2 = load_nasdaq(use_fed_data=False,use_dif=use_dif)

    input_dim = 1+use_dif
    model_name = 'GRU'
    study_name = 'study_gru_no_fed'

    study_gru_no_fed = conduct_study(study_name, model_name,input_dim,scaled, n_trials)
    joblib.dump(study_gru_no_fed, os.path.join(res_dir, study_name + end))

    ### LSTM without Fed hyperparameter search
    model_name = 'LSTM'
    study_name = 'study_lstm_no_fed'
    study_lstm_no_fed = conduct_study(study_name, model_name,input_dim,scaled, n_trials)
    joblib.dump(study_lstm_no_fed, os.path.join(res_dir, study_name + end))

if __name__ == "__main__":
    hyperparameter_tuning()