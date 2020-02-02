import os, sys
import numpy
import time
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.externals import joblib
from train_svm import *
import matplotlib.pyplot as plt

def test_svm(train_fn, test_fn, testdata_type):
  start_time = time.time()
  train_directory = os.getcwd() + '\\' + train_fn + '\\'
  test_directory = os.getcwd() + '\\' + test_fn + '\\'
  svm = joblib.load(train_directory + train_fn + '_svm_train/' + train_fn + '_svm_train.pkl') 

  #change Xmat to appro. suffix
  X_test = numpy.loadtxt(test_directory + test_fn + '_'+ testdata_type +'.txt', skiprows=1, usecols = (1,2,3,4,5))   #if mat already constructed
  pred = svm.predict(X_test)

  y_test = numpy.loadtxt(test_directory + test_fn + '_'+ testdata_type +'.txt', skiprows=1, usecols = (6,))
  acc = round(accuracy_score(y_test, pred), 3)
  print('Train: ', train_fn, '   Test: ', test_fn, '   Accuracy: ', acc)

  fpr, tpr, thresholds = roc_curve(y_test, pred)
  auc_score = round(auc(fpr, tpr), 3)
  plt.figure()
  plt.plot(fpr, tpr)
  plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  output_dir = train_directory + '\\test_outputs\\'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  plt.savefig(output_dir + train_fn +'_'+ test_fn + '_ROC.png')
  #plt.show()
  print('Train: ', train_fn, '   Test: ', test_fn, '   AUC: ', auc_score)
  
  node_pairs = numpy.genfromtxt(test_directory + test_fn + '_'+ testdata_type +'.txt', skip_header=1, usecols = (0,),dtype=None, encoding=None)
  pred_fn = output_dir + 'pred_' + test_fn + '_' + testdata_type + '.txt'
  pred = pred.tolist()
  y_test = y_test.tolist()
  with open(pred_fn,'w') as f:
    f.write('Accuracy: ' + str(acc) + '\n')
    f.write('AUC: ' + str(auc_score) + '\n')
    num_zeros = pred.count(0)
    num_ones = len(pred) - num_zeros
    f.write('Pred, Number of 0s: ' + str(num_zeros) + '\n')
    f.write('Pred, Number of 1s: ' + str(num_ones) + '\n')

    num_zeros = y_test.count(0)
    num_ones = len(y_test) - num_zeros
    f.write('Actual, Number of 0s: ' + str(num_zeros) + '\n')
    f.write('Actual, Number of 1s: ' + str(num_ones) + '\n\n')

    f.write('Classified wrong:\n')
    f.write('node_pairs, pred, actual\n')
    for i in range(len(pred)):
      if int(pred[i]) != int(y_test[i]):
        f.write(str(node_pairs[i]) + ', ' + str(pred[i]) + ', ' + str(y_test[i]) + '\n')
    f.write("--- %s seconds ---" % (time.time() - start_time))
  return acc, auc_score
