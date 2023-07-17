import sys,os,time
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial.distance import pdist, squareform

def kernel(x,epsi):
    return np.exp ( -(x/epsi)**2)

def euclidian_norm(P1,P2,T1,T2,EGR1,EGR2,PHI1,PHI2):
    distP = ( P1 - P2 )**2
    distT = ( T1 - T2 )**2
    distEGR = ( EGR1 - EGR2 )**2
    distPHI = ( PHI1 - PHI2 )**2
    return  np.sqrt( distEGR + distP + distPHI + distT )

def compute_errors(s_init,S):
    errors = [np.abs((s - S[i])/S[i]).tolist() for i,s in enumerate(s_init)][0]
    max_errors = [np.max(err) for err in errors]
    max_errors_index = [np.argmax(err) for err in errors]
    return errors,max_errors,max_errors_index

def compute_all_distances(inputs):

    st = time.time()
    distances = squareform(pdist(inputs.values, 'euclidean'))
    et = time.time()
    print("Compute all euclidian norms time : %2.2f s" % (et-st))

    return distances


def gradient_descent(inputs,outputs,training_index,init_epsilon_k,errors_epsi_1,slope=1e-15):
    
    nb_index = len(training_index)
    nb_val_table = inputs.shape[0]

    #ref = np.zeros(nb_val_table)
    ref = errors_epsi_1
    S = outputs.values.T

    # Epsilons initialization
    epsi_1 = np.ones( nb_index )* init_epsilon_k
    epsi_2 = epsi_1 * 1.01

    # Initialise a second point for GD
    rbf = Rbf(*(*inputs.loc[training_index].values.T , outputs.loc[training_index].values),
                      epsilon = epsi_2, function='gaussian', norm='euclidean', mode='N-D',
                      )
    s_init = rbf(*inputs.values.T).T.reshape(S.shape) #fixing wk to find the best epsilon

    errors_epsi_2,max_errors_epsi_2,_ = compute_errors(s_init,S) #errors to be minimized modifing epsilon
 
    ite_dg=0
    erreur=1
    while erreur > 0 :
        ite_dg += 1 
        s = np.zeros(nb_val_table)
        grad = np.zeros(nb_index)

        #compute the gradient to be minimized and fill the new epsilon table
        #f(epsi)=errors => grad(f)= (f(epsi+delta)-f(epsi))/delta
        for k in range(nb_index) :
            epsi_diff = epsi_2[k] - epsi_1[k]
            if epsi_diff == 0 :
                grad[k] = 0
            else:
                grad[k]=(errors_epsi_2[k] - errors_epsi_1[k])/epsi_diff
                if grad[k]*slope > epsi_diff :
                    grad[k] = epsi_diff/slope

        if np.max(max_errors_epsi_2) > np.max(errors_epsi_1):
            epsi_2[:] = epsi_1[:] - grad[:]*slope
        else:
            epsi_1[:] = epsi_2[:]
            epsi_2[:] = epsi_1[:] - grad[:]*slope
            errors_epsi_1[:] = errors_epsi_2[:]

        # compute the new RBF parameters with the new epsilon vector
        rbf = Rbf(*(*inputs.loc[training_index].values.T , outputs.loc[training_index].values),
                        epsilon = epsi_2, function='gaussian', norm='euclidean', mode='N-D',
                        )

        # compute the interpolation with the new epsilon vector and updated wk
        s = rbf(*inputs.values.T).T.reshape(S.shape) #fixing wk to find the best epsilon

        # compute the new error to be minimized with the new epsilon vector
        errors_epsi_2,max_errors_epsi_2,_ = compute_errors(s,S)

        erreur = np.max(max_errors_epsi_2) - np.max(ref)
        
        #print("Ref max error is %3.2f" % (np.max(abs(ref)) *100), "%")
        print("At ite ",ite_dg," maximum error is %3.2f" % (np.max(max_errors_epsi_2) *100), "%")

    return errors_epsi_2, np.max(max_errors_epsi_2), epsi_2, rbf.nodes


        # # compute the new RBF parameters with the new epsilon vector
        # for i in range(nb_index):
        #     for j in range(nb_index):                
        #         matriceA[i,j] =  kernel(dist_all[training_index[i],training_index[j]],epsi_2[j]) 

        # # recompute the new wk with the new epsilon vector
        # w = np.linalg.solve(matriceA,S_training)
        
        # # compute the interpolation with the new epsilon vector and updated wk
        # for i in range(nb_val_table):
        #     for k in range(nb_index):
        #         s[i] += w[k] * kernel(dist_all[i,training_index[k]],epsi_2[k])

        #s = s.reshape(S.shape)    

        # compute the new error
        #errors_epsi_2,max_errors,max_errors_index = compute_errors(s,S)