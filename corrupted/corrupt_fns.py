import math, random, time, os, pickle, sys, networkx, numpy
from operator import itemgetter
import operator
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, roc_curve, auc
import sim_fns

def adjlist_to_adjmat(graph_dict):
  n = len(graph_dict)
  adj_mat = numpy.zeros((n,n))
  for node in graph_dict:
    neighbors = graph_dict[node]
    for neigh in neighbors:
      adj_mat[node, neigh] = 1
  return adj_mat

def minthree(x, y, z):
   minimum = x
   if y < minimum:
      minimum = y
   if z < minimum:
      minimum = z
   return minimum

def corrupt(netw, samp, P):
  graph_dict = pickle.load( open(netw + '_graph.p', "rb" ) )
  graph_netx = networkx.to_networkx_graph(graph_dict)
  K = math.floor(graph_netx.number_of_edges() * P) #number of edges to remove
  rand_edges = random.sample(graph_netx.edges(), int(K)) #edges to remove
  graph_netx.remove_edges_from(rand_edges)
  graph_dict_corr = networkx.to_dict_of_lists(graph_netx)  #back to adj list b/c faster when assessing neighbors of pairs
  return rand_edges, graph_dict_corr

#use this func after creating the graph from corrupt_netw.py
def make_comb_testmat(P, samp, rand_edges, graph_dict, degsep_test_pairs):
  adj_mat = adjlist_to_adjmat(graph_dict)
  #calculate summed even powers of adj matrix to find which node pairs are within path of len K away
  test = numpy.dot(adj_mat,adj_mat)
  sums = test #sums includes G^2 to G^(k+2)
  kmax = 3
  sp_name = 'sp_'+str(kmax)
  for i in range(1):  #range(k) means sums includes G^2 to G^(k+2)
    test = numpy.dot(test,adj_mat) 
    sums = sums + test
  for i in range(len(sums)):
    sums[i,i] = 0  #set diag as 0 b/c no self loops for link prediction

  all_sums = adj_mat + sums #all degr < kmax, not just from 2 to kmax
  for i in range(len(all_sums)):
    all_sums[i,i] = 1  #set diag as 1 b/c no self loops for link prediction

  #combine removed edges and pairs in 2 to 3 degs of sep that were not in training set
  data_mat = []
  node_pairs = [] #keep track of which node_pairs have already been analyzed
  for node_pair in degsep_test_pairs:
    node_1 = node_pair[0]
    node_2 = node_pair[1]
    if (node_1,node_2) not in node_pairs or (node_2, node_1) not in node_pairs: #b/c undirected graph
      node_pairs.append((node_1,node_2))
      common_neigh = sim_fns.common_neigh(graph_dict, node_1, node_2)
      jaccard = sim_fns.jaccard(graph_dict, node_1, node_2)
      pref_attach = sim_fns.pref_attach(graph_dict, node_1, node_2)
      adamic = sim_fns.adamic(graph_dict, node_1, node_2)
      sp = sums[node_1, node_2] #all shortest paths up to length kmax b/w the 2 nodes

      data_row = [common_neigh, jaccard, pref_attach, adamic, sp, 0]  #all degsep pairs have no paths of len 1
      data_mat.append(data_row)

  for node_pair in rand_edges:
    node_1 = node_pair[0]
    node_2 = node_pair[1]
    if (node_1,node_2) not in node_pairs or (node_2, node_1) not in node_pairs: #b/c undirected graph
      node_pairs.append((node_1,node_2))
      common_neigh = sim_fns.common_neigh(graph_dict, node_1, node_2)
      jaccard = sim_fns.jaccard(graph_dict, node_1, node_2)
      pref_attach = sim_fns.pref_attach(graph_dict, node_1, node_2)
      adamic = sim_fns.adamic(graph_dict, node_1, node_2)
      sp = sums[node_1, node_2] #all shortest paths up to length kmax b/w the 2 nodes

      data_row = [common_neigh, jaccard, pref_attach, adamic, sp, 1]   #all rmvd edges nodes were originally class 1
      data_mat.append(data_row)

  test_node_pairs = degsep_test_pairs + rand_edges
  return data_mat

def construct_feature_mat(graph_dict, rand_edges):
  adj_mat = adjlist_to_adjmat(graph_dict)
  #calculate summed even powers of adj matrix to find which node pairs are within path of len K away

  test = numpy.dot(adj_mat,adj_mat)
  sums = test #sums includes G^2 to G^(k+2)
  kmax = 3
  sp_name = 'sp_'+str(kmax)
  for i in range(1):  #range(k) means sums includes G^2 to G^(k+2)
    test = numpy.dot(test,adj_mat) 
    sums = sums + test
  for i in range(len(sums)):
    sums[i,i] = 0  #set diag as 0 b/c no self loops for link prediction
  
  pairs_linked = numpy.transpose(numpy.nonzero(adj_mat))

  all_sums = adj_mat + sums #all degr < kmax, not just from 2 to kmax
  for i in range(len(all_sums)):
    all_sums[i,i] = 1  #set diag as 1 b/c no self loops for link prediction
  #don't use rmvd edges in training, since they will be used for testing
  for pair in rand_edges:
    sums[pair[0],pair[1]] = 1
  pairs_not_linked = numpy.transpose(numpy.where(all_sums == 0))
  #create pairs not linked before pairs within degr b/c you will artifically alter sums using rand edges

  #don't use rmvd edges in training, since they will be used for testing
  for pair in rand_edges:
    sums[pair[0],pair[1]] = 0
  pairs_within_degr = numpy.transpose(numpy.nonzero(sums))

  #find min len of (linked,within,not_linked) and use it as the total # of training samples
  num_train = minthree(len(pairs_linked), len(pairs_within_degr), len(pairs_not_linked))
  if (num_train % 2) != 0:
    num_train += 1 #make it even so you can divide it into balanced training set

  #assume len(pairs_linked) >= (num_train/2)
  train_pairs = numpy.concatenate((pairs_linked[0:(num_train//2)],pairs_within_degr[0:(num_train//4)]))
  train_pairs = numpy.concatenate((train_pairs,pairs_not_linked[0:(num_train//4)]))

  #test data of class 0 but within 2 to 3 degr of sep. Used to combine w/ rmvd to get FP
  degsep_test_pairs = pairs_within_degr[0:(num_train//4)]
  degsep_test_pairs = degsep_test_pairs.tolist()

  data_mat = []
  node_pairs = {} #keep track of which node_pairs have already been analyzed
  for node_pair in train_pairs:
    node_1 = node_pair[0]
    node_2 = node_pair[1]
    if (node_1,node_2) not in node_pairs or (node_2, node_1) not in node_pairs: #b/c undirected graph
      common_neigh = sim_fns.common_neigh(graph_dict, node_1, node_2)
      jaccard = sim_fns.jaccard(graph_dict, node_1, node_2)
      pref_attach = sim_fns.pref_attach(graph_dict, node_1, node_2)
      adamic = sim_fns.adamic(graph_dict, node_1, node_2)
      sp = sums[node_1, node_2] #all shortest paths up to length kmax b/w the 2 nodes

      data_row = [common_neigh, jaccard, pref_attach, adamic, sp]   
      if node_2 in graph_dict[node_1]:
        link_class = 1
      else:
        link_class = 0
      data_row.append(link_class)
      data_mat.append(data_row)
      node_pairs[(node_1,node_2)] = data_row  #read this into .txt after done looping

  X = [row[0:-1] for row in data_mat] #data_mat does not contain node pair names
  y = [row[-1] for row in data_mat]
  return X, y, degsep_test_pairs

def svm_train(X,y):
  start_time = time.time()
  C = 10
  gamma = 1e-3
  clf = svm.SVC(C=C, gamma=gamma)
  clf.fit(X, y)  
  return clf

def test_svm_corrupt(clf, test_mat, train_X, train_y):
  X_test = [row[0:-1] for row in test_mat] #data_mat does not contain node pair names
  test_pred = clf.predict(X_test)

  y_test = [row[-1] for row in test_mat]
  test_acc = accuracy_score(y_test, test_pred)

  fpr, tpr, thresholds = roc_curve(y_test, test_pred)
  test_auc_score = auc(fpr, tpr)

  train_pred = clf.predict(train_X)
  train_acc = accuracy_score(train_y, train_pred)

  fpr, tpr, thresholds = roc_curve(train_y, train_pred)
  train_auc_score = auc(fpr, tpr)

  return [train_acc, test_acc, train_auc_score, test_auc_score]