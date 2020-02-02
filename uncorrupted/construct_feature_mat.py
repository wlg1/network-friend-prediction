import time, os, pickle, operator
from operator import itemgetter
import numpy
from sklearn import svm
from sklearn.externals import joblib
import sim_fns

def make_list(filename,input_dict):
  output = open(filename,'w')
  for head in input_dict: 
    row = str(head) + ' : ' + str(input_dict[head])
    output.write(row + '\n')
  output.close()

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

def make_graph_csv(filename, graph_dict):
  csv = open(filename,"w")
  for v in graph_dict: 
      row=str(v)
      edges = graph_dict[v]
      for t in edges:
          row += ';' + str(t)
      csv.write(row + '\n')
  csv.close()

def make_graph(header):
  f = open(header +'.gdf', 'r')  #gdf not included due to containing private user info, so must start with .p and .csv
  graph_dict = {}
  edges_signal = False #when true, time to read in edges. 
  i=0 #new node id. Same as index id
  node_to_id = {} #links node original id (str) to new node id (int)
  id_to_node = {}
  for line in f:
    #if i > 10000: break #due to memory issues
    splitted = line.split(',')
    if edges_signal == False: #read in nodes
      if splitted[0].isdigit() == True:
        id_to_node[i] = splitted[0].replace('\n','')
        node_to_id[splitted[0].replace('\n','')] = i
        graph_dict[i] = []
        i += 1
      if splitted[0] == 'edgedef>node1 VARCHAR': #before this line, file has nodes. after it, file has edges
        edges_signal = True
    #after reading in nodes, read in edges
    else:
      head = node_to_id[splitted[0].replace('\n','')]
      tail = node_to_id[splitted[1].replace('\n','')]
      graph_dict[head].append(tail)
      graph_dict[tail].append(head) #edges in gdf are undirected

  filename = header + '_graph.p'
  with open(filename, 'wb') as f:
    pickle.dump(graph_dict, f)

  sorted_id_to_node = sorted(id_to_node.items(), key=operator.itemgetter(0))
  output = header + '_node_id.txt'
  make_list(output,id_to_node)

  make_graph_csv(header + '_graph.csv', graph_dict) #for use in gephi
  return graph_dict

def construct_feature_mat(input_fn):
  directory = os.getcwd() + '\\' + input_fn   #use \\ b/c / may sometimes be for unicode
  if not os.path.exists(directory):
    os.makedirs(directory)
  header = directory + '\\' + input_fn 
  if os.path.isfile(header+'_graph.p'):
    graph_dict = pickle.load( open(header + '_graph.p', "rb" ) )
  else:
    graph_dict = make_graph(input_fn)

  make_graph_csv(header + '_graph_ids.csv', graph_dict)
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
  pairs_within_degr = numpy.transpose(numpy.nonzero(sums))

  all_sums = adj_mat + sums #all degr < kmax, not just from 2 to kmax
  for i in range(len(all_sums)):
    all_sums[i,i] = 1  #set diag as 1 b/c no self loops for link prediction
  pairs_not_linked = numpy.transpose(numpy.where(all_sums == 0))

  #find min len of (linked,within,not_linked) and use it as the total # of training samples
  num_train = minthree(len(pairs_linked), len(pairs_within_degr), len(pairs_not_linked))
  if (num_train % 2) != 0:
    num_train += 1 #make it even so you can divide it into balanced training set

  #assume len(pairs_linked) >= (num_train/2)
  train_pairs = numpy.concatenate((pairs_linked[0:(num_train//2)],pairs_within_degr[0:(num_train//4)]))
  train_pairs = numpy.concatenate((train_pairs,pairs_not_linked[0:(num_train//4)]))

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
      node_pairs[(node_1,node_2)] = data_row

  #data_mat = sorted(data_mat, key=itemgetter(-1))
  X = [row[0:-1] for row in data_mat] #data_mat does not contain node pair names
  y = [row[-1] for row in data_mat]

  #sort data_mat by class first (optional), then extract X and y, THEN write to txt 
  #all data is in feature_matrix. separate data acc to the dist b/w node pairs (1 degr away, b/w 2 and Kmax, and >Kmax)
  #this separation is used to test how well the classifier does on different types of node pairs
  feature_matrix = open(header + '_Xmat.txt', 'w') #save it as txt so it can be fed into other programs like R, not just pickle for python
  linked_mat = open(header + '_linked.txt', 'w')
  degsep_mat = open(header + '_degsep.txt', 'w')
  far_mat = open(header + '_veryfar.txt', 'w')
  feature_matrix.write('pair common_neigh jaccard pref_attach adamic ' + sp_name + ' class\n')
  linked_mat.write('pair common_neigh jaccard pref_attach adamic ' + sp_name + ' class\n')
  degsep_mat.write('pair common_neigh jaccard pref_attach adamic ' + sp_name + ' class\n')
  far_mat.write('pair common_neigh jaccard pref_attach adamic ' + sp_name + ' class\n')

  pairs_linked = pairs_linked.tolist()
  pairs_within_degr = pairs_within_degr.tolist()
  pairs_not_linked = pairs_not_linked.tolist()
  for node_pair in node_pairs:
    data_row = node_pairs[node_pair]
    node_1 = node_pair[0]
    node_2 = node_pair[1]
    write_row = str(node_1) + '-' + str(node_2)
    for stat in data_row: 
      write_row += ' ' + str(stat)
    feature_matrix.write(write_row + '\n')
    if [node_1,node_2] in pairs_linked: 
      linked_mat.write(write_row + '\n')
    elif [node_1,node_2] in pairs_within_degr: 
      degsep_mat.write(write_row + '\n')
    else:
      far_mat.write(write_row + '\n')
