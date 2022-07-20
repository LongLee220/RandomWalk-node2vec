import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

from scipy import spatial

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=4,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
    
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walk_lol = []
	for walk in walks:
	    tmp = []
	    for node in walk:
	        tmp.append(str(node))
	    walk_lol.append(walk)
	model = Word2Vec(walk_lol, vector_size=args.dimensions, window=args.window_size, min_count=1, sg=1, workers=args.workers, epochs=args.iter) 
	model.wv.save_word2vec_format(args.output)
	
	return model


nx_G = nx.Graph()

# the metrix of medic&medic
medic1 = []
medic2 = []
vals = []
# f = open('/Users/longlee/Documents/Code/Drug/adj_dcc.csv','r')
# f=open('/Users/longlee/Downloads/Gowalla_edges.txt')
f = open("/Users/longlee/Documents/Code/K_Core/0.edges", "r")

# t = 0
# while True:
    
#     line = f.readline()
#     tmp = line.strip().split(',')#把分隔号抹去
    
#     for num in range(len(tmp)):
        
#         if tmp[num] == '1':
#             nx_G.add_edge(int(t), int(num),weight=1)
            
#             medic1.append(int(t))
#             medic2.append(int(num))
#             vals.append(int(1))
#     nx_G.add_node(int(t))#孤立点162,233,241,285,489,550,699,782
#     t = t+1
        
#     if not line:
#         break
# f.close()
 

while True:
    
    line = f.readline()
    tmp = line.strip().split(' ')#把分隔号抹去 ', '\t'去大空格
    if len(tmp) >1:
        
        nx_G.add_edge(int(tmp[0]), int(tmp[1]),weight=1)
    
    
    if not line:
        break
    
f.close()

Nodes = list(nx_G.nodes())
N = max(Nodes)
for node in range(N):
    if node not in nx_G.nodes():
        nx_G.add_node(node)


args = parse_args()
# nx_G = read_graph()
G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
G.preprocess_transition_probs()
walks = G.simulate_walks(args.num_walks, args.walk_length)
model = learn_embeddings(walks)


# print(nx.is_connected(nx_G))
# print(model.wv[1])
# print(model.wv.similarity(1,2))
# print(model.wv.most_similar(1))

def cos_similarity(v_1,v_2):
    return 1-spatial.distance.cosine(v_1,v_2)

# print(cos_similarity(model.wv[1],model.wv[2]))


embedding_node = []

Nodes = list(nx_G.nodes())
N = max(Nodes)
for ide in range(N):
    w = Nodes[ide]
    embedding_node.append(model.wv[w])
embedding_noded = np.matrix(embedding_node).reshape(N,-1)
y_pred = cluster.KMeans(n_clusters=9,random_state=9).fit_predict(embedding_node)
print(y_pred)


colors = ['pink', 'blue', 'green', 'yellow', 'red', 'brown','teal','aqua','darkblue','black']
Num_col = set(y_pred)
Node_col = {}
for node in Nodes:
    col_ide = y_pred[node-1] 
    Node_col[node] = colors[col_ide]


    
# 运用布局
# pos=nx.circular_layout(G)          # 生成圆形节点布局
# pos=nx.random_layout(G)            # 生成随机节点布局
# pos=nx.shell_layout(G)             # 生成同心圆节点布局
# pos=nx.spring_layout(G)            # 利用Fruchterman-Reingold force-directed算法生成节点布局
# pos=nx.spectral_layout(G)          # 利用图拉普拉斯特征向量生成节点布局
# pos=nx.kamada_kawai_layout(G)      #使用Kamada-Kawai路径长度代价函数生成布局


pos = nx.spring_layout(nx_G)

nx.draw_networkx_nodes(nx_G, pos, node_size=10, cmap=plt.cm.RdYlBu, node_color=list(Node_col.values()))
nx.draw_networkx_edges(nx_G, pos, width=0.5,alpha=1)

# nx.draw(nx_G, pos=pos, node_size=100, width=2, node_color=list(Node_col.values()))
plt.savefig("/Users/longlee/Documents/Code/Node2vec/cluuster.png")
plt.show(nx_G)

