import pdb
import math

def common_neigh(graph_dict, node_1, node_2):
  a = graph_dict[node_1]
  b = graph_dict[node_2]
  return len(list(set(a) & set(b)))

def jaccard(graph_dict, node_1, node_2):
  a = graph_dict[node_1]
  b = graph_dict[node_2]
  if len(list(set(a) | set(b))) > 0:
    return float(len(list(set(a) & set(b)))) / len(list(set(a) | set(b)))  #must use float or else it returns 0
  else:
    return 0

def pref_attach(graph_dict, node_1, node_2):
  a = graph_dict[node_1]
  b = graph_dict[node_2]
  return len(a) * len(b)

def adamic(graph_dict, node_1, node_2):
  a = graph_dict[node_1]
  b = graph_dict[node_2]
  inter = list(set(a) & set(b))
  total = 0
  for node in inter:
    freq = math.log(len(graph_dict[node])) #base e
    if freq != 0:
      total += 1/freq
  return total



