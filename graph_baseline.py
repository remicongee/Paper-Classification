import networkx as nx
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression

# Create a directed graph
G = nx.read_edgelist('Data/Cit-HepTh.txt', delimiter='\t', create_using=nx.DiGraph())

print("Nodes: ", G.number_of_nodes())
print("Edges: ", G.number_of_edges())

# Read training data
train_ids = list()
y_train = list()
with open('Data/train.csv', 'r') as f:
    next(f)
    for line in f:
        t = line.split(',')
        train_ids.append(t[0])
        y_train.append(t[1][:-1])

n_train = len(train_ids)
unique = np.unique(y_train)
print("\nNumber of classes: ", unique.size)

# Create the training matrix. Each row corresponds to an article.
# Use the following 3 features for each article:
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_ids)
X_train = np.zeros((n_train, 3))
for i in range(n_train):
	X_train[i,0] = G.out_degree(train_ids[i])
	X_train[i,1] = G.in_degree(train_ids[i])
	X_train[i,2] = avg_neig_deg[train_ids[i]]


# Read test data
test_ids = list()
with open('Data/test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Create the test matrix. Use the same 3 features as above
n_test = len(test_ids)
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_ids)
X_test = np.zeros((n_test, 3))
for i in range(n_test):
	X_test[i,0] = G.out_degree(test_ids[i])
	X_test[i,1] = G.in_degree(test_ids[i])
	X_test[i,2] = avg_neig_deg[test_ids[i]]


print("\nTrain matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the articles of the test set
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('Result/graph_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)