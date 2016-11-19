import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mp
import random as rd
import argparse
import os, sys
import csv
import math
import time
import matplotlib.pyplot as pl

class County:
    def __init__(self,parsename):
        self.parsename = parsename
        dataframe = xls.parse(parsename)
        self.date = dataframe['Date']
        self.hour = dataframe['Hr_End']
        self.demand = dataframe['RT_Demand']
        self.drybulb = dataframe['Dry_Bulb']
        self.dewpnt = dataframe['Dew_Point']
    def disp_all(self):
        print self.dataframe
    def get_all(self):
        return self.dataframe

xls = pd.ExcelFile('smd_hourly.xls')
INC = County('ISO NE CA')
ME = County('ME')
NH = County('NH')
VT = County('VT')
CT = County('CT')
RI = County('RI')
SEMA = County('SEMA')
WCMA = County('WCMA')
NEMA = County('NEMA')

data_dir = './data/' # directory contains input data
num_epoches = 50000 # training epoches for each customer samples
day_steps = 24
n_steps = day_steps # input size
test_batch_size = 4*7*day_steps # days of a batch
validation_batch_size = 0*day_steps
train_batch_size = 7*day_steps
feature_size = 1 # same time of a week
n_hidden = 10 # input size
num_layers = 1
n_output = 1
totalen = np.array(INC.demand).shape[0]

# DEMAND MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
tmp = np.array(INC.demand)
demand_mat = tmp.reshape([1,tmp.shape[0],1])
demand_mat = demand_mat/25000
#demand_mat = np.concatenate([demand_mat,np.array(ME.demand).reshape([1,np.array(ME.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(NH.demand).reshape([1,np.array(NH.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(VT.demand).reshape([1,np.array(VT.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(CT.demand).reshape([1,np.array(CT.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(RI.demand).reshape([1,np.array(RI.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(SEMA.demand).reshape([1,np.array(SEMA.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(WCMA.demand).reshape([1,np.array(WCMA.demand).shape[0],1])],axis = 0)
#demand_mat = np.concatenate([demand_mat,np.array(NEMA.demand).reshape([1,np.array(NEMA.demand).shape[0],1])],axis = 0)
print demand_mat.shape
# DRY BULB MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
tmp = np.array(INC.drybulb)
drybulb_mat = tmp.reshape([1,tmp.shape[0],1])
drybulb_mat = drybulb_mat/100
#drybulb_mat = np.concatenate([drybulb_mat,np.array(ME.drybulb).reshape([1,np.array(ME.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(NH.drybulb).reshape([1,np.array(NH.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(VT.drybulb).reshape([1,np.array(VT.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(CT.drybulb).reshape([1,np.array(CT.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(RI.drybulb).reshape([1,np.array(RI.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(SEMA.drybulb).reshape([1,np.array(SEMA.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(WCMA.drybulb).reshape([1,np.array(WCMA.drybulb).shape[0],1])],axis = 0)
#drybulb_mat = np.concatenate([drybulb_mat,np.array(NEMA.drybulb).reshape([1,np.array(NEMA.drybulb).shape[0],1])],axis = 0)
#print drybulb_mat.shape
# DEW PNT MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
tmp = np.array(INC.dewpnt)
dewpnt_mat = tmp.reshape([1,tmp.shape[0],1])
dewpnt_mat = dewpnt_mat/100
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(ME.dewpnt).reshape([1,np.array(ME.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(NH.dewpnt).reshape([1,np.array(NH.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(VT.dewpnt).reshape([1,np.array(VT.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(CT.dewpnt).reshape([1,np.array(CT.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(RI.dewpnt).reshape([1,np.array(RI.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(SEMA.dewpnt).reshape([1,np.array(SEMA.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(WCMA.dewpnt).reshape([1,np.array(WCMA.dewpnt).shape[0],1])],axis = 0)
#dewpnt_mat = np.concatenate([dewpnt_mat,np.array(NEMA.dewpnt).reshape([1,np.array(NEMA.dewpnt).shape[0],1])],axis = 0)
#print dewpnt_mat.shape

#db = np.concatenate([demand_mat,dewpnt_mat,drybulb_mat],axis = 2)
#db = np.concatenate([demand_mat,dewpnt_mat],axis = 2)
db = demand_mat

print db.shape

#define id arrays
test_id = np.array(test_batch_size)
valid_id = np.array(validation_batch_size)
train_id = np.array(totalen-test_batch_size-validation_batch_size-n_steps)

#give values to id arrays
rang = range(n_steps,totalen-test_batch_size)
valid_id = rd.sample(rang,validation_batch_size)
test_id = np.array(range(totalen-test_batch_size,totalen))
train_id = set(range(n_steps,totalen-test_batch_size))-set(valid_id)

#sort three id array
valid_id = np.sort(valid_id)
test_id = np.sort(test_id)
train_id = np.array(list(train_id))

def train_data_gen():
    X = np.zeros((train_batch_size,n_steps,feature_size))
    Y = np.zeros((train_batch_size,feature_size))
    count = 0
    rang = range(n_steps,train_id.shape[0])
    train_rd = rd.sample(rang,train_batch_size)
    train_rd = np.sort(train_rd)
    for i in train_rd:
        Y[count] = db[:,i,:]
        X[count] = db[:,i-n_steps:i,:]
        count = count + 1
    return (X,Y)

def valid_data_gen():
    X = np.zeros((train_batch_size,n_steps,feature_size))
    Y = np.zeros((train_batch_size,feature_size))
    count = 0
    rang = range(n_steps,valid_id.shape[0])
    valid_rd = rd.sample(rang,train_batch_size)
    valid_rd = np.sort(valid_rd)
    for i in valid_rd:
        Y[count] = db[:,i,:]
        X[count] = db[:,i-n_steps:i,:]
        count = count + 1
    return (X,Y)

def test_data_gen():
    X = np.zeros((test_batch_size,n_steps,feature_size))
    Y = np.zeros((test_batch_size,feature_size))
    count = 0
    for i in test_id:
        Y[count] = db[:,i,:]
        X[count] = db[:,i-n_steps:i,:]
        count = count + 1
    return (X,Y)

# create placeholder for x and y
#with tf.device('/gpu:0'):
x = tf.placeholder("float",[None,n_steps,feature_size])
istate = tf.placeholder("float",[None,num_layers*2*n_hidden])
y = tf.placeholder("float",[None,n_output])


# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([feature_size, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}
def RNN(_X, _istate, _weights, _biases):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, feature_size]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(stacked_lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)
#cost function 
cost = tf.reduce_mean(tf.pow(pred[:,0]-y[:,0],2)) # cost function of this batch of data
#compute parameter updates
#optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

def maxe(predictions, targets):
    return max(abs(predictions-targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mape(predictions, targets):
    return np.mean(abs(predictions-targets)/targets)

outlist = np.zeros([(num_epoches/10),test_batch_size])
kind = 0
time1 = time.time()
# generate test data
test_x,test_y = test_data_gen()
test_x = test_x.reshape(test_batch_size,n_steps,feature_size)
print test_x
print test_y
### Execute
# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Create a summary to monitor cost function
    #tf.scalar_summary("loss", cost)
    # Merge all summaries to a single operator
    #merged_summary_op = tf.merge_all_summaries()

    # tensorboard info.# Set logs writer into folder /tmp/tensorflow_logs
    #summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph_def=sess.graph_def)

    #initialize all variables in the model
    sess.run(init)
    for k in range(num_epoches):
        #Generate Data for each epoch
        #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        #this is required to feed data into rnn.rnn
        #print traindays
        X,Y = train_data_gen()
        X = X.reshape(train_batch_size,n_steps,feature_size)


        #Create the dictionary of inputs to feed into sess.run
        #if k < 0:
        #    sess.run(optimizer2,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})
        #else:
        sess.run(optimizer,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})   
        #perform an update on the parameters
        #cost1 = sess.run(cost,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
        #print "Iter " + str(k) + ", Minibatch Loss ---- Train = " + str(cost1)
        # Write logs at every iteration
        #if k>50 & k%10 == 0:
        #    summary_str = sess.run(merged_summary_op, feed_dict={x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
        #    summary_writer.add_summary(summary_str, k)

        #if k % 10 == 0:
        if k % 10 == 0:
            #print test_x
            output_tmp_ex = sess.run(pred,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )  
            print "Iter " + str(k) + " ---- Process: " + "{:.2f}".format(100*float(k)/float(num_epoches)) + "%"
            outp_test = output_tmp_ex[:,0]
            #print outp_test[0:3]
            outlist[kind,:] = outp_test.copy().T
            kind = kind + 1
        #    print ktmp
        #if k % 10 == 0:
        #    output_tmp_ex = sess.run(pred,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
        #    print "Iter " + str(k) + ", Minibatch Loss ---- Train = " + "{:.6f}".format(cost) # + "; Test = " + "{:.6f}".format(loss2)
    #print "haha{}".format(outp)
        #    ktmp = np.corrcoef(output_tmp_ex.T,test_y.T)[0,1]
        #    accuracy1.append(ktmp)
        #    print ktmp

RList = np.zeros([(num_epoches/10)])
rmseList = np.zeros([(num_epoches/10)])
maxeList = np.zeros([(num_epoches/10)])
mapeList = np.zeros([(num_epoches/10)])
for i in range(kind):
    out = np.array(outlist[i])
    tmp = out.T.reshape((1,test_batch_size))
    RList[i] = np.corrcoef(tmp[0,:],test_y.T[0,:])[0,1]
    rmseList[i] = rmse(tmp[0,:],test_y.T[0,:])
    maxeList[i] = maxe(tmp[0,:],test_y.T[0,:])
    mapeList[i] = mape(tmp[0,:],test_y.T[0,:])

prefix = './gefcom-result/INC'
postfix = '-' + str(num_layers) + '-' + str(n_hidden) + '.csv'
#DataFrame(RList).to_csv(prefix + 'R' + postfix)
#DataFrame(rmseList).to_csv(prefix + 'RMSE' + postfix)
#DataFrame(maxeList).to_csv(prefix + 'MAXE' + postfix)
#DataFrame(mapeList).to_csv(prefix + 'MAPE' + postfix)
DataFrame(out).to_csv(prefix + 'out3.csv')
DataFrame(test_y).to_csv(prefix + 'test3.csv')
time2 = time.time()
print time2-time1

