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
# Naive predictor
We compared our models to a naïve predictor which predicts next week’s closing value
based on the average weekly gain – g of the Nasdaq:

![image](https://user-images.githubusercontent.com/66019798/124433542-88c10880-dd7b-11eb-9f53-d8ab60e99ae6.png)
# File name	Purpose
| File Name        | Purpose           |
| ---------------- |:-----------------:|
| Prediction_FED.ipynb | Main code file including all the projects results results  |
| optuna_search.py | Hyper parameters tuning    |
| utils.py         | Loading the data, declaring project's models,models training, declaring obective for optuna and printing functions  | 
| studies          | Optunas' studies results |
| input            | The data we used for our project which includes Nasdaq-100 index values and the FED assets|  
# Final results for every model 
We shall denote, t - the target closing price, p - our prediction for the closing price.
The loss over a batch will be the MAPE – mean absolute percentage error:

![image](https://user-images.githubusercontent.com/66019798/124433596-970f2480-dd7b-11eb-90f4-c2772ec72570.png)

And so:

![image](https://user-images.githubusercontent.com/66019798/124432375-3af7d080-dd7a-11eb-8da1-7ceffee9b390.png)
# Best Model's results 


![image](https://user-images.githubusercontent.com/66019798/124432568-71355000-dd7a-11eb-834c-96cbf28bab94.png)

![image](https://user-images.githubusercontent.com/66019798/124432738-9e81fe00-dd7a-11eb-85ae-1c48f23bcd8a.png)

Future prediction (when the last data dates back to June 23rd):

![image](https://user-images.githubusercontent.com/66019798/124432762-a346b200-dd7a-11eb-8d00-22dc167efd4f.png)

# DATA Sources
Nasdaq 100-values:
https://finance.yahoo.com/quote/%5ENDX/history?period1=958608000&period2=1621296000&interval=1wk&filter=history&frequency=1wk&includeAdjustedClose=true

Fed total assets:
https://fred.stlouisfed.org/series/WALCL

# References
Based on the work "Predicting Stock Price using LSTM model, PyTorch" by Taron Zakaryan, Paris, Île-de-France, France.

Project site can be found here: https://www.kaggle.com/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch
