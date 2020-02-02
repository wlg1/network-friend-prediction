import pdb, traceback, sys
import operator, time, math, os, random, pickle
import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from corrupt_fns import *
from heatmap import *

def plot_score_vs_rmvd(metric, scores):
  plt.clf()
  plt.plot(numpy.linspace(0.1, 0.9, 9), scores, 'ro')
  plt.title('PCT_RMVD vs ' + metric)
  plt.xlabel('Percentage of edges removed')
  plt.ylabel('Avg Test ' + metric)
  plt.savefig('avg_test_'+metric+'.png')

def main():
  lst_avg_test_acc = []
  lst_avg_test_AUC = []
  netw = 'network_A'
  score_names = ['train_acc', 'test_acc', 'train_AUC', 'test_AUC']
  total_iter = 100
  acc_mat = numpy.zeros((9, 2))
  AUC_mat = numpy.zeros((9, 2))
  for i, P in enumerate(numpy.linspace(0.1, 0.9, 9)): # % of edges to remove
    start_time = time.time()
    P = round(P, 1)
    lst_score_sums = [0,0,0,0] 
    for samp in range(1, total_iter + 1):
      print(P, samp)
      rand_edges, graph_dict = corrupt(netw, samp, P)
      train_X, train_y, degsep_test_pairs = construct_feature_mat(graph_dict, rand_edges)
      clf = svm_train(train_X, train_y) 
      test_mat = make_comb_testmat(str(P), samp, rand_edges, graph_dict, degsep_test_pairs)
      test_results = test_svm_corrupt(clf, test_mat, train_X, train_y)
      lst_score_sums = list(map(operator.add, lst_score_sums, test_results))
    lst_score_avgs = [round(float(score) / total_iter, 3) for score in lst_score_sums]
    lst_avg_test_acc.append(lst_score_avgs[1])
    lst_avg_test_AUC.append(lst_score_avgs[3])
    for j, score in enumerate(lst_score_avgs[0:2]):
      acc_mat[i, j] = score
    for j, score in enumerate(lst_score_avgs[2:4]):
      AUC_mat[i, j] = score
    with open(netw + '_corr_' + str(P) + '_avg_pred.txt','w') as f:
      for name, score in zip(score_names, lst_score_avgs):
        f.write('Avg ' + name + ' for ' + str(total_iter) +' iterations: ' + str(score)+'\n')
      f.write("--- %s seconds ---" % (time.time() - start_time))
  plot_score_vs_rmvd('accuracy', lst_avg_test_acc)
  plot_score_vs_rmvd('AUC', lst_avg_test_AUC)
  x_catg = [round(P, 1) for P in numpy.linspace(0.1, 0.9, 9)]
  make_heatmap(acc_mat, x_catg, ['Avg train acc', 'Avg test acc'], 'Accuracy')
  make_heatmap(AUC_mat, x_catg, ['Avg train AUC', 'Avg test AUC'], 'AUC')

if __name__ == '__main__':
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
