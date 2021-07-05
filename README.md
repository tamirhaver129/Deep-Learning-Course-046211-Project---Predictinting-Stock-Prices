# Deep Learning Course 046211 Project: Predictinting NASDAQ-100 using LSTM&GRU and Federal Reserve data
#
Python implementation for NASDAQ-100 price prediction using FED's assets.

Project has been done by Roy Mahlab & Tamir Haver.
![image](https://user-images.githubusercontent.com/66019798/123539089-ac170280-d740-11eb-8720-c61655529faa.png)


# Tools 

In this project we coded in the python using Jupyter Notebook.

When runnig the code on Jupyter one should run it cell by cell in the same order as it is brought in our repository.

# Agenda
The Federal Reserve System (also known as the Federal Reserve or simply the Fed) is the central banking system of the United States of America.

Since the corona-virus pandemic broke, the United States Federal Reserve 
assets nearly doubled, which raised claims that the Fed is inflating the value of assets in the stock market.

In our project we will examine this claim. We will do so by implementing 2 different methods for RNN models, and compare their results with
each other and examine the influence of the FED's assets on the results.

We can divide the workflow into few main steps:

Loading the data

Spliting the data

Building the LSTM and GRU models

Using OPTUNA for tuning of hyper parameters

Compering the results and the naive prediction (average weekly gain)

Presenting the results

# Parameters
The hyper parameters we used OPTUNA for tuning are:

Batch Size

Hidden and Output Dim of GRU and LSTM

N layers of GRU and LSTM

Learning Rate

Look Back – How many past weeks do we need in order to predict next week’s value.
# File name	Purpsoe
ls_dqn_main.py	general purpose main application for training/playing a LS-DQN agent
pong_ls_dqn.py	main application tailored for Atari's Pong
boxing_ls_dqn.py	main application tailored for Atari's Boxing
dqn_play.py	sample code for playing a game, also in ls_dqn_main.py
actions.py	classes for actions selection (argmax, epsilon greedy)
agent.py	agent class, holds the network, action selector and current state
dqn_model.py	DQN classes, neural networks structures
experience.py	Replay Buffer classes
hyperparameters.py	hyperparameters for several Atari games, used as a baseline
srl_algorithms.py	Shallow RL algorithms, LS-UPDATE
utils.py	utility functions
wrappers.py	DeepMind's wrappers for the Atari environments
*.pth	Checkpoint files for the Agents (playing/continual learning)
Deep_RL_Shallow_Updates_for_Deep_Reinforcement_Learning.pdf	Writeup - theory and results

# DATA Sources
Nasdaq 100-values:
https://finance.yahoo.com/quote/%5ENDX/history?period1=958608000&period2=1621296000&interval=1wk&filter=history&frequency=1wk&includeAdjustedClose=true

Fed total assets:
https://fred.stlouisfed.org/series/WALCL

# References
Based on the work "Predicting Stock Price using LSTM model, PyTorch" by Taron Zakaryan, Paris, Île-de-France, France.

Project site can be found here: https://www.kaggle.com/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch
