import sys,os,time
import pandas as pd
import numpy as np
import random
from scipy.interpolate import Rbf
from grad_desc import gradient_descent,compute_errors

debug = False

#np.random.seed(377)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
path = os.getcwd()

files=[
    '../data/table.csv',
]

files=[path+file for file in files]
data=pd.concat([pd.read_csv(file) for file in files])
inputs = data[['P','Tin','phi','EGR']]
outputs = data[['u']]

# devide each columns by its max value to normalize the data and avoid ill matrix to be inverted
inputs = inputs.apply(lambda x: x/np.max(x))

S = outputs.values.T
# print("S shape:", S.shape)

# build a training set of 50 samples and a test set of the rest
training_index = np.random.choice(inputs.index, size=50, replace=False)
#data_test = inputs.drop(training_index, axis=0)#.reset_index(drop=True)
#S_test = outputs.drop(training_index, axis=0)#.reset_index(drop=True)

#Training
rbf = Rbf(*(*inputs.loc[training_index].values.T, outputs.loc[training_index,]), 
          function='gaussian', norm='euclidean', mode='N-D')

#Test
s_init = rbf(*inputs.values.T).T.reshape(S.shape)

if debug:
    print("s_init shape:", s_init.shape)
    print("S shape:", S.shape)

#compute the error between the initial s and the real s for each column
#the error variable is a list of list containing the error for each column
error,max_error,index = compute_errors(s_init,S)


ultimate_error = np.max(max_error)
ultimate_index = index[np.argmax(max_error)]
if debug:
    print("Initial ultimate error:", ultimate_error)
    print("Initial ultimate index:", ultimate_index)

epsilon_k = rbf.epsilon

while ultimate_error > 0.03:
    additional_index = [ultimate_index]
    training_index = np.concatenate([training_index, additional_index], axis=0)
    #data_test = data_test.drop(additional_index, axis=0)#.reset_index(drop=True)
    #S_test = S_test.drop(additional_index, axis=0)#.reset_index(drop=True)

    #Since a training point is added, we need to recompute the epsilons and add one to the list
    epsilon_k = np.append(epsilon_k, np.mean(epsilon_k))

    #Training
    rbf = Rbf(*(*inputs.loc[training_index].values.T , outputs.loc[training_index,]), 
              function='gaussian', norm='euclidean', mode='N-D', epsilon=epsilon_k)

    #Test
    interp = rbf(*inputs.values.T).T.reshape(S.shape)

    # Error between the interpolated points and the real S for each column
    error_epsilon_rbf, max_errors, _= compute_errors(interp,S)
    ultimate_error = np.max(max_errors)

    # loop on the number of column in outputs
    #for i,output in enumerate(outputs.columns):
    if ultimate_error > 0.05:
        print("Max error after training: %3.2f" % (ultimate_error * 100), "%")
        ultimate_index = np.argmax(error_epsilon_rbf)
    else:
        print("______________")
        print("Max error after training, before GD: %3.2f" % (ultimate_error * 100), "%")
        error_epsilon_rbf,ultimate_error, epsilon_k, w_k = gradient_descent(inputs,outputs,training_index,rbf.epsilon,error_epsilon_rbf)
        print("Max error after training, after GD: %3.2f" % (ultimate_error * 100), "%")
        ultimate_index = np.argmax(error_epsilon_rbf)
        print("______________")

    print("number of training points:", len(training_index))