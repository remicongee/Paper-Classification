from utils import *
from model import *


filename = 'Data/trainText_nltk.txt'
train_text = list()
with open(filename, 'r') as file_to_read:
    lines = file_to_read.readlines()
    for line in lines:
        train_text.append(line)

filename = 'Data/testText_nltk.txt'
test_text = list()
with open(filename, 'r') as file_to_read:
    lines = file_to_read.readlines()
    for line in lines:
        test_text.append(line)

text_train, extractor = extractFeature(train_text)
text_test = extractor.transform(test_text).toarray()

train_path = 'Data/train.csv'
test_path = 'Data/test.csv'
train_ids, y_train = readIdLabel(train_path)
unique = np.unique(y_train).tolist()
index = range(len(unique))
index2label = dict(zip(index, unique))
label2index = dict(zip(unique, index))
test_ids, _ = readIdLabel(test_path)

graph_cite_path = 'Data/Cit-HepTh.txt'
info_path = 'Data/node_information.csv'
info_df = readInfo(info_path)

n_train = len(train_ids)
G = readGraph(graph_cite_path)
UG = nx.Graph(G)
nxc = nx.clustering(UG)
avg_neig_deg_in = nx.average_neighbor_degree(G, nodes=train_ids, source='in', target='in')
avg_neig_deg_out = nx.average_neighbor_degree(G, nodes=train_ids, source='out', target='out')
graph_train = np.zeros((n_train, 6))
for i in range(n_train):
	degree_abs = G.out_degree(train_ids[i]) + G.in_degree(train_ids[i]) + 1e-6
	graph_train[i,0] = G.out_degree(train_ids[i]) / degree_abs
	graph_train[i,1] = G.in_degree(train_ids[i]) / degree_abs
	degree_abs = avg_neig_deg_in[train_ids[i]] + avg_neig_deg_out[train_ids[i]] + 1e-6
	graph_train[i,2] = avg_neig_deg_in[train_ids[i]] / degree_abs
	graph_train[i,3] = avg_neig_deg_out[train_ids[i]] / degree_abs
	graph_train[i,4] = nxc[train_ids[i]] * 10
	graph_train[i,5] = (float(info_df.loc[info_df['id'] == int(train_ids[i])]['year'].iloc[0]) - 1992.0) / (2000.0 - 1992.0)

n_test = len(test_ids)
avg_neig_deg_in = nx.average_neighbor_degree(G, nodes=test_ids, source='in', target='in')
avg_neig_deg_out = nx.average_neighbor_degree(G, nodes=test_ids, source='out', target='out')
graph_test = np.zeros((n_test, 6))
for i in range(n_test):
	degree_abs = G.out_degree(test_ids[i]) + G.in_degree(test_ids[i]) + 1e-6
	graph_test[i,0] = G.out_degree(test_ids[i]) / degree_abs
	graph_test[i,1] = G.in_degree(test_ids[i]) / degree_abs
	degree_abs = avg_neig_deg_in[test_ids[i]] + avg_neig_deg_out[test_ids[i]] + 1e-6
	graph_test[i,2] = avg_neig_deg_in[test_ids[i]] / degree_abs
	graph_test[i,3] = avg_neig_deg_out[test_ids[i]] / degree_abs
	graph_test[i,4] = nxc[test_ids[i]] * 10
	graph_test[i,5] = (float(info_df.loc[info_df['id'] == int(test_ids[i])]['year'].iloc[0]) - 1992.0) / (2000.0 - 1992.0)

cite_train = np.zeros((n_train, len(unique)))
for i, id in enumerate(train_ids):
    for node in G.successors(id):
        if node in train_ids:
            node = y_train[train_ids.index(node)]
            cite_train[i][label2index[node]] += 1

cite_test = np.zeros((n_test, len(unique)))
for i, id in enumerate(test_ids):
    for node in G.successors(id):
        if node in train_ids:
            node = y_train[train_ids.index(node)]
            cite_test[i][label2index[node]] += 1

graph_cited_path = 'Data/Cited-HepTh.txt'
GG = readGraph(graph_cited_path)
cited_train = np.zeros((n_train, len(unique)))
for i, id in enumerate(train_ids):
    for node in GG.successors(id):
        if node in train_ids:
            node = y_train[train_ids.index(node)]
            cited_train[i][label2index[node]] += 1

cited_test = np.zeros((n_test, len(unique)))
for i, id in enumerate(test_ids):
    for node in GG.successors(id):
        if node in train_ids:
            node = y_train[train_ids.index(node)]
            cited_test[i][label2index[node]] += 1

x_train = list()
for i in range(n_train):
    feature_train = np.hstack((cite_train[i] + cited_train[i], graph_train[i], text_train[i]))    
    x_train.append(feature_train)

x_test = list()
for i in range(n_test):
    feature_test = np.hstack((cite_test[i] + cited_test[i], graph_test[i], text_test[i]))
    x_test.append(feature_test)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
clf = LogisticRegression(max_iter=1000, tol=1e-4, C=2.0, penalty='l2')
clf.fit(x_train, y_train)

y_pred = clf.predict_proba(x_test)
# Write predictions to a file
with open('Result/submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)

''' 
####### Pytorch Implementation ######
input_size = len(x_train[0])
epoch = int(50)
learning_rate = 0.001

clf = Classifier(input_size=input_size)
clf = torch.nn.DataParallel(clf).cuda()
print(clf)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

train(model=clf, epoch=epoch, optimizer=optimizer, features=x_train, labels=y_train, criterion=criterion, label_dict=label2index)

y_pred = test(model=clf, features=x_test)
y_pred = np.reshape(np.array(y_pred), (-1, len(unique)))

np.save("Result/Submission.npy", y_pred)

# Write predictions to a file
with open('Result/submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    unique.insert(0, "Article")
    writer.writerow(unique)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
'''


