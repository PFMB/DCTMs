#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr  2 16:56:45 2021

Python 3.8.1 with packages as in requirements.txt
https://github.com/siboehm/NormalizingFlowNetwork cloned at 1st of April '21

2h runtime on one standard CPU
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from estimators import *
from evaluation.scorers import mle_log_likelihood_score, DummySklearWrapper
import pandas as pd

tfd = tfp.distributions
#np.random.seed(22)

# indices taken from R to foster comparability between algorithms
p_data = "" # set wd accordingly
sp_lit = pd.read_csv(p_data + "bench_split.csv", header='infer',index_col=0)

pls_res = [] 
reps = 20 # variation for std. dev. of PLS
iters = 1000 # epochs
    
for no_flows in range(1,6):
    ### airfoil
    
    air = [i.strip().split() for i in open(p_data + "airfoil/airfoil_self_noise.dat").readlines()]
    air = pd.DataFrame(air)
    air.index = np.arange(1, len(air) + 1) # force 1 index (instead of 0) for test/train split
    
    air_train = np.array(sp_lit["air_train"])
    air_train = air_train[~np.isnan(air_train)].astype(int)
    air_test = np.array(sp_lit["air_test"])
    air_test = air_test[~np.isnan(air_test)].astype(int)
    
    air_train = air[~air.index.isin(air_train)]
    air_test = air[~air.index.isin(air_test)]
    
    n_row = air_train.shape[0]
    air_x_train = np.float32(air_train[air_train.columns[0:4]]).reshape(n_row,4)
    air_y_train = np.float32(air_train[air_train.columns[5]]).reshape(n_row,1)
    
    n_row = air_test.shape[0]
    air_x_test = np.float32(air_test[air_test.columns[0:4]]).reshape(n_row,4)
    air_y_test = np.float32(air_test[air_test.columns[5]]).reshape(n_row,1)
    
    pls_air = [] 
    for idx in range(reps):
        # may want to put #tf.random.set_seed(random_seed), in line 13 of BaseEstimator.py to provoke model variance
        model = NormalizingFlowNetwork(n_dims=1,
                                       random_seed= 1 + idx,
                                       trainable_base_dist=False,
                                       n_flows = no_flows,
                                       learning_rate = 1e-3, 
                                       hidden_sizes=(8, 8))
        # put #callbacks=[tf.keras.callbacks.TerminateOnNaN()], in line 29 of BaseEstimator.py to avoid: 
        #   fit() got multiple values for keyword argument 'callbacks'
        model.fit(air_x_train, air_y_train, 
                  epochs=iters, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=100, verbose=2)],
                  validation_split=0.2,
                  verbose=2)
        pls_air.append(mle_log_likelihood_score(DummySklearWrapper(model), air_x_test, air_y_test))
    
    pls_air = np.array(pls_air)

    
    ### diabetes
    
    diab_y_train = pd.read_csv(p_data + "diabetes/y_train_diabetes.csv", header=None)
    diab_x_train = pd.read_csv(p_data + "diabetes/x_train_diabetes.csv", header=None)
    n_row = diab_y_train.shape[0]
    diab_y_train = np.float32(diab_y_train).reshape(n_row,1)
    diab_x_train = np.float32(diab_x_train).reshape(n_row,10)
    
    diab_x_test = pd.read_csv(p_data + "diabetes/x_test_diabetes.csv", header=None)
    diab_y_test = pd.read_csv(p_data + "diabetes/y_test_diabetes.csv", header=None)
    n_row = diab_y_test.shape[0]
    diab_y_test = np.float32(diab_y_test).reshape(n_row,1)
    diab_x_test = np.float32(diab_x_test).reshape(n_row,10)
    
    
    pls_diab = [] 
    for idx in range(reps):
        # may want to put #tf.random.set_seed(random_seed), in line 13 of BaseEstimator.py to provoke model variance
        model = NormalizingFlowNetwork(n_dims=1,
                                       random_seed= 1 + idx,
                                       trainable_base_dist=False,
                                       n_flows = no_flows,
                                       learning_rate = 1e-3, 
                                       hidden_sizes=(8, 8))
        # put #callbacks=[tf.keras.callbacks.TerminateOnNaN()], in line 29 of BaseEstimator.py to avoid: 
        #   fit() got multiple values for keyword argument 'callbacks'
        model.fit(diab_x_train, diab_y_train, 
                  epochs=iters, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=100, verbose=2)],
                  validation_split=0.2,
                  verbose=2)
        pls_diab.append(mle_log_likelihood_score(DummySklearWrapper(model), diab_x_test, diab_y_test))
    
    pls_diab = np.array(pls_diab)
    
    
    ### forest fire
    
    ff = pd.read_csv(p_data + "forestfires/forestfires.csv", header=0)
    ff = pd.DataFrame(ff)
    ff.index = np.arange(1, len(ff) + 1) # force 1 index (instead of 0) for test/train split
    ff["area"] = np.log(ff["area"] + 1) # cf. descrip. https://archive.ics.uci.edu/ml/datasets/forest+fires
    
    # one way to handle categoricals is dummies
    #ff = pd.concat([ff, pd.get_dummies(ff["month"])], axis=1)
    #del ff["month"]
    #ff = pd.concat([ff, pd.get_dummies(ff["day"])], axis=1)
    #del ff["day"] 
    
    # another way is as int (gives highest PLS)
    ff["month"] = pd.factorize(ff["month"])[0]
    ff["day"] = pd.factorize(ff["day"])[0]
    
    ff_train = np.array(sp_lit["fire_train"])
    ff_train = ff_train[~np.isnan(ff_train)].astype(int)
    ff_test = np.array(sp_lit["fire_test"])
    ff_test = ff_test[~np.isnan(ff_test)].astype(int)
    
    ff_train = ff[~ff.index.isin(ff_train)]
    ff_test = ff[~ff.index.isin(ff_test)]
    
    n_row = ff_train.shape[0]
    ff_x_train = np.float32(ff_train.drop(["area"], axis=1)).reshape(n_row,12) #29 with dummy
    ff_y_train = np.float32(ff_train["area"]).reshape(n_row,1)
    
    n_row = ff_test.shape[0]
    ff_x_test = np.float32(ff_test.drop(["area"], axis=1)).reshape(n_row,12)
    ff_y_test = np.float32(ff_test["area"]).reshape(n_row,1)
    
    pls_ff = [] 
    for idx in range(reps):
        # may want to put #tf.random.set_seed(random_seed), in line 13 of BaseEstimator.py to provoke model variance
        model = NormalizingFlowNetwork(n_dims=1,
                                       random_seed= 1 + idx,
                                       trainable_base_dist=False,
                                       n_flows = no_flows,
                                       learning_rate = 1e-3, 
                                       hidden_sizes=(8, 8))
        # put #callbacks=[tf.keras.callbacks.TerminateOnNaN()], in line 29 of BaseEstimator.py to avoid: 
        #   fit() got multiple values for keyword argument 'callbacks'
        model.fit(ff_x_train, ff_y_train, 
                  epochs=iters, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=100, verbose=2)],
                  validation_split=0.2,
                  verbose=2)
        pls_ff.append(mle_log_likelihood_score(DummySklearWrapper(model), ff_x_test, ff_y_test))
    
    pls_ff = np.array(pls_ff)

    
    ### boston
    
    bost_y_train = pd.read_csv(p_data + "boston/y_train.csv", header=None)
    bost_x_train = pd.read_csv(p_data + "boston/x_train.csv", header=None)
    n_row = bost_y_train.shape[0]
    bost_y_train = np.float32(bost_y_train).reshape(n_row,1)
    bost_x_train = np.float32(bost_x_train).reshape(n_row,13)
    
    bost_x_test = pd.read_csv(p_data + "boston/x_test.csv", header=None)
    bost_y_test = pd.read_csv(p_data + "boston/y_test.csv", header=None)
    n_row = bost_y_test.shape[0]
    bost_y_test = np.float32(bost_y_test).reshape(n_row,1)
    bost_x_test = np.float32(bost_x_test).reshape(n_row,13)
    
    pls_bost = [] 
    for idx in range(reps):
        # may want to put #tf.random.set_seed(random_seed), in line 13 of BaseEstimator.py to provoke model variance
        model = NormalizingFlowNetwork(n_dims=1,
                                       random_seed= 1 + idx,
                                       trainable_base_dist=False,
                                       n_flows = no_flows,
                                       learning_rate = 1e-3, 
                                       hidden_sizes=(8, 8))
        # put #callbacks=[tf.keras.callbacks.TerminateOnNaN()], in line 29 of BaseEstimator.py to avoid: 
        #   fit() got multiple values for keyword argument 'callbacks'
        model.fit(bost_x_train, bost_y_train, 
                  epochs=iters, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=100, verbose=2)],
                  validation_split=0.2,
                  verbose=2)
        pls_bost.append(mle_log_likelihood_score(DummySklearWrapper(model), bost_x_test, bost_y_test))
    
    pls_bost = np.array(pls_bost)
    
    air = {"Airfoil Mean": "{:.2f}".format(pls_air.mean()), "Airfoil StdDev": "{:.2f}".format(pls_air.std())}
    bost = {"Boston Mean": "{:.2f}".format(pls_bost.mean()), "Boston StdDev": "{:.2f}".format(pls_bost.std())}
    diab = {"Diabetes Mean": "{:.2f}".format(pls_diab.mean()), "Diabetes StdDev": "{:.2f}".format(pls_diab.std())}
    ff = {"ForestFires Mean": "{:.2f}".format(pls_ff.mean()), "ForestFires StdDev": "{:.2f}".format(pls_ff.std())}

    pls_res.append([air,bost,diab,ff])

pls_res
