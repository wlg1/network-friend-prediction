import pdb, traceback, sys
import os, time, operator, pickle
from operator import itemgetter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.externals import joblib

import sim_fns
from construct_feature_mat import *
from train_svm import *
from test_svm import *
from heatmap import *

def main():
  ### CONSTRUCT FEATURE MATRICES ###
  fb_networks = ['network_A', 'network_B']
  for netw in fb_networks:
    construct_feature_mat(netw)
  #make_graph('yt')
  yt_networks = ['ytbig', 'ytsmall']
  for netw in yt_networks:
    construct_feature_mat(netw)

  ### TRAINING ###
  networks = fb_networks + yt_networks
  for netw in networks:
    svm_train(netw)

  ### TESTING ###
  acc_mat = np.zeros((4, 4))
  AUC_mat = np.zeros((4, 4))
  testdata_type = 'Xmat'  #IE) Xmat, degsep, linked, veryfar
  for i, train_netw in enumerate(networks):
    if not os.path.exists(train_netw + '\\test_outputs\\'):
      os.makedirs(train_netw + '\\test_outputs\\')
    f = open(train_netw + '\\test_outputs\\' + train_netw + '_summary.txt','w')
    for j, test_netw in enumerate(networks):
      acc, AUC = test_svm(train_netw, test_netw, testdata_type)
      acc_mat[i, j] = acc
      AUC_mat[i, j] = AUC
      f.write(test_netw + ' Accuracy: ' + str(acc) + '\n')
      f.write(test_netw + ' AUC: ' + str(AUC) + '\n')
  make_heatmap(acc_mat, networks, networks, 'Accuracy')
  make_heatmap(AUC_mat, networks, networks, 'AUC')

if __name__ == '__main__':
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)