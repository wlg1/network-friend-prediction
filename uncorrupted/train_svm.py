import sim_fns
import operator
from sklearn import svm
from sklearn.externals import joblib
import time
import numpy
import os
from operator import itemgetter
import pickle

def svm_train(input_fn):    #feature mat already stored
  header = os.getcwd() + '\\' + input_fn + '\\' + input_fn
  X = numpy.loadtxt(header + '_Xmat.txt', skiprows=1, usecols = (1,2,3,4,5))
  y = numpy.loadtxt(header + '_Xmat.txt', skiprows=1, usecols = (6,))
  start_time = time.time()
  C = 10
  gamma = 1e-3
  clf = svm.SVC(C=C, gamma=gamma)
  #clf = svm.SVC(kernel=’linear’)
  clf.fit(X, y)  
  print("--- %s seconds ---" % (time.time() - start_time))
  directory = header + '_svm_train' + '\\'
  if not os.path.exists(directory):
    os.makedirs(directory)
  joblib.dump(clf, directory + input_fn + '_svm_train.pkl') #manually create folder

  """
  # if using linear kernel
  importances = clf.feature_importances_
  indices = numpy.argsort(importances)[::-1]

  print("Feature ranking:")
  for f in range(X.shape[1]):
      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
  """
