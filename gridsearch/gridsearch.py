import pdb, traceback, sys
import os, sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def get_opt_param(input_fn):
  header = os.getcwd() + '\\' + input_fn
  X = np.loadtxt(header + '_Xmat.txt', skiprows=1, usecols = (1,2,3,4,5))
  y = np.loadtxt(header + '_Xmat.txt', skiprows=1, usecols = (6,))
  #X = StandardScaler().fit_transform(X)

  C_range = 10. ** np.arange(-3, 4)
  gamma_range = 10. ** np.arange(-5, 4)
  param_grid = dict(gamma=gamma_range, C=C_range)
  cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
  grid.fit(X, y)  #takes a while

  print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
  #print("The best classifier is: ", grid.best_estimator_)

  score_dict = grid.grid_scores_  #parameter settings and scores
  scores = [x[1] for x in score_dict]
  scores = np.array(scores).reshape(len(C_range), len(gamma_range))
  plt.figure(figsize=(8, 6))
  plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(scores, interpolation='nearest')
  plt.xlabel('gamma')
  plt.ylabel('C')
  plt.colorbar()
  plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
  plt.yticks(np.arange(len(C_range)), C_range)
  plt.savefig(input_fn+'_heatmap.png')
  plt.show()

def main():
  get_opt_param('network_A')
  #get_opt_param('network_B')
  #get_opt_param('youtube')

if __name__ == '__main__':
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
