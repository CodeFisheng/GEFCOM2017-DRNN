import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
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
num_epoches = 20000# training epoches for each customer samples
input_seq_size = 7*24 # input size
test_batch_size = 3 # days of a batch
valid_batch_size = 14
train_batch_size = 3
data_dim = 1 # same time of a week
output_seq_size = 24
totalen = np.array(INC.demand).shape[0]/output_seq_size
n_hidden = 5 # input size
num_layers = 2

# DEMAND MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
tmp = np.array(INC.demand, dtype = np.float32)
demand_mat = tmp.reshape([tmp.shape[0]/output_seq_size,output_seq_size,1])
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
tmp = np.array(INC.drybulb, dtype = np.float32)
drybulb_mat = tmp.reshape([tmp.shape[0]/output_seq_size,output_seq_size,1])
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
tmp = np.array(INC.dewpnt, dtype = np.float32)
dewpnt_mat = tmp.reshape([tmp.shape[0]/output_seq_size,output_seq_size,1])
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
print db

#define id arrays
test_id = np.array(test_batch_size)
valid_id = np.array(valid_batch_size)
train_id = np.array(totalen-test_batch_size-valid_batch_size-input_seq_size)

#give values to id arrays
rang = range(input_seq_size/output_seq_size,totalen-test_batch_size)
valid_id = rd.sample(rang,valid_batch_size)
test_id = np.array(range(totalen-test_batch_size,totalen))
train_id = set(range(input_seq_size/output_seq_size,totalen-test_batch_size))-set(valid_id)

#sort three id array
valid_id = np.sort(valid_id)
test_id = np.sort(test_id)
train_id = np.array(list(train_id))
print valid_id
print test_id
print train_id

def train_data_gen():
    X = np.zeros((input_seq_size,train_batch_size,data_dim))
    Y = np.zeros((output_seq_size,train_batch_size,data_dim))
    count = 0
    rang = range(input_seq_size/output_seq_size,train_id.shape[0])
    train_rd = rd.sample(rang,train_batch_size)
    train_rd = np.sort(train_rd)
    for i in train_rd:
        Y[:,count,:] = db[i,:,:]
        X[:,count,:] = (db[i-input_seq_size/output_seq_size:i,:,:]).reshape([input_seq_size,data_dim])
        count = count + 1
    return (X,Y)

def valid_data_gen():
    X = np.zeros((input_seq_size,train_batch_size,data_dim))
    Y = np.zeros((output_seq_size,train_batch_size,data_dim))
    count = 0
    rang = range(input_seq_size/output_seq_size,valid_id.shape[0])
    valid_rd = rd.sample(rang,train_batch_size)
    valid_rd = np.sort(valid_rd)
    for i in valid_rd:
        Y[:,count,:] = db[i,:,:]
        X[:,count,:] = (db[i-input_seq_size/output_seq_size:i,:,:]).reshape([input_seq_size,data_dim])
        count = count + 1
    return (X,Y)

def test_data_gen():
    X = np.zeros((input_seq_size,test_batch_size,data_dim))
    Y = np.zeros((output_seq_size,test_batch_size,data_dim))
    count = 0
    for i in test_id:
        Y[:,count,:] = db[i,:,:]
        X[:,count,:] = (db[i-input_seq_size/output_seq_size:i,:,:]).reshape([input_seq_size,data_dim])
        count = count + 1
    return (X,Y)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
        
_X = [tf.placeholder(tf.float32, shape = [train_batch_size, data_dim]) for _ in xrange(input_seq_size)]
_Y = [tf.placeholder(tf.float32, shape = [train_batch_size, data_dim]) for _ in xrange(output_seq_size)]
#_X = tf.reshape(_X, [-1, data_dim])
#_Y = tf.reshape(_Y, [-1, data_dim])
weights_in = [ tf.Variable(tf.random_normal([data_dim, n_hidden])) for _ in xrange(input_seq_size) ]
bias_in = [ tf.Variable(tf.random_normal([n_hidden])) for _ in xrange(input_seq_size) ]

weights_out = [ tf.Variable(tf.random_normal([n_hidden, data_dim])) for _ in xrange(output_seq_size)]
bias_out = [ tf.Variable(tf.random_normal([data_dim])) for _ in xrange(output_seq_size)]

print _X[0]
encoder_inputs = [ tf.matmul(_X[i], weights_in[i]) + bias_in[i] for i in xrange(input_seq_size) ]
decoder_inputs = [ tf.matmul(_Y[i], weights_in[i]) + bias_in[i] for i in xrange(output_seq_size) ]

cell = rnn_cell.GRUCell(n_hidden)
dropout = tf.constant(0.75, dtype = tf.float32)
cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout)
cell = rnn_cell.MultiRNNCell([cell]*num_layers)

model_outputs, states = seq2seq.basic_rnn_seq2seq(_X, _Y, cell)

_pred = [ tf.matmul(model_outputs[i], weights_out[i]) + bias_out[i] for i in xrange(output_seq_size)]

reshaped_outputs = tf.reshape(_pred, [-1])
reshaped_results = tf.reshape(_Y, [-1])

cost = tf.reduce_mean(tf.pow(reshaped_outputs-reshaped_results,2))

variable_summaries(cost, 'cost')
#compute parameter updates
#optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.001, 0.8, 0.7).minimize(cost)

def maxe(predictions, targets):
    return np.max(abs(predictions-targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mape(predictions, targets):
    return np.mean(abs(predictions-targets)/targets)

outlist = np.zeros([(num_epoches/10),output_seq_size*test_batch_size])
kind = 0
time1 = time.time()
# generate test data
test_x,test_y = test_data_gen()
test_y_unknown = np.zeros((output_seq_size,test_batch_size,data_dim)) # test if it can use for prediction
test_x = test_x.astype(np.float32)
test_y = test_y.astype(np.float32)
tex_list = {key: value for (key, value) in zip(_X, test_x)}
tey_list = {key: value for (key, value) in zip(_Y, test_y_unknown)}
### Execute
# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Create a summary to monitor cost function
    #tf.scalar_summary("loss", cost)
    # Merge all summaries to a single operator
    merged_summary_op = tf.merge_all_summaries()

    # tensorboard info.# Set logs writer into folder /tmp/tensorflow_logs
    summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph_def=sess.graph)

    #initialize all variables in the model
    sess.run(init)
    costs = []
    costs_test = []
    for k in range(num_epoches):
        #Generate Data for each epoch
        #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        #this is required to feed data into rnn.rnn
        #print traindays
        X,Y = train_data_gen()
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        #Create the dictionary of inputs to feed into sess.run
        #if k < 0:
        #    sess.run(optimizer2,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})
        #else:
        x_list = {key: value for (key, value) in zip(_X, X)}
        y_list = {key: value for (key, value) in zip(_Y, Y)}
        summary, err, _ = sess.run([merged_summary_op, cost, optimizer], feed_dict=dict(x_list.items() + y_list.items()))
        costs.append(err)
        #perform an update on the parameters
        #cost1 = sess.run(cost,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
        if k%10==0:
            print "Iter " + str(k) + ", Minibatch Loss ---- Train = " + str(err)
            err_test,pred_test = sess.run([cost, reshaped_outputs], feed_dict=dict(tex_list.items() + tey_list.items()))
            costs_test.append(err_test)
            outlist[kind,:] = pred_test.copy().T
            kind = kind + 1
time2 = time.time()
print time2-time1
costlist = np.array(costs)
costlist_te = np.array(costs_test)
prefix = './seq2seq/'
DataFrame(costlist).to_csv(prefix + 'costfile_train.csv')
DataFrame(costlist_te).to_csv(prefix + 'costfile_test.csv')
outlist = outlist * 25000
actuaload = np.reshape(test_y, [-1])
actuaload = (actuaload*25000).T
prediction = np.array(outlist[-1,:])
DataFrame(actuaload).to_csv(prefix + 'actual_load.csv')
DataFrame(prediction).to_csv(prefix + 'prediction.csv')

RList = np.zeros([(num_epoches/10)])
rmseList = np.zeros([(num_epoches/10)])
maxeList = np.zeros([(num_epoches/10)])
mapeList = np.zeros([(num_epoches/10)])
actuaload = actuaload.reshape((1,test_batch_size*output_seq_size))
for i in range(kind):
    out = np.array(outlist[i])
    tmp = out.T.reshape((1,test_batch_size*output_seq_size))
    RList[i] = np.corrcoef(tmp,actuaload)[0,1]
    rmseList[i] = rmse(tmp,actuaload)
    maxeList[i] = maxe(tmp,actuaload)
    mapeList[i] = mape(tmp,actuaload)
DataFrame(RList).to_csv(prefix + 'R.csv')
DataFrame(rmseList).to_csv(prefix + 'RMSE.csv')
DataFrame(maxeList).to_csv(prefix + 'MAXE.csv')
DataFrame(mapeList).to_csv(prefix + 'MAPE.csv')
plt.plot(costs)
plt.show()
