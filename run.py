from model import Classifier
from utils import *
import torch


# path initialization
graph_cite_path = 'Data/Cit-HepTh.txt'
graph_cited_path = 'Data/Cited-HepTh.txt'
train_path = 'Data/train.csv'
test_path = 'Data/test.csv'
info_path = 'Data/node_information'

# construct graph dataframe
graph_cite = readGraph(graph_cite_path)
graph_cited = readGraph(graph_cited_path)

# read node information
info_df = readInfo(info_path)

# read training set, make training graph
x_train_id, y_train_label = readIdLabel(train_path)
graph_cite_train = graphTrain(graph_cite, x_train_id)
graph_cited_train = graphTrain(graph_cited, x_train_id)

# transform label to one-hot vector
y_train_onehot = label2onehot(y_train_label)

# read test set
x_test_id, _ = readIdLabel(test_path)

# read text
x_train_text = readText(info_df, x_train_id)
x_test_text = readText(info_df, x_test_id)

# extract features, make graph featured
x_train_feature, extractor = extractFeature(x_train_text)
x_test_feature = extractor.transform(x_test_text)

# make citation feature
# for training set
x_train_cite = combineFeature(graph_cite_train, x_train_id, x_train_feature)
x_train_cited = combineFeature(graph_cited_train, x_train_id, x_train_feature)
# for whole data
x_test_cite = combineFeature(graph_cite, x_test_id, x_test_feature)
x_test_cited = combineFeature(graph_cited, x_test_id, x_test_feature)

# initialization
feature_size = len(x_train_cite[0])
clf_feat = Classifier(input_size=feature_size, mode='feature')
clf_cite = Classifier(input_size=feature_size, mode='cite')
clf_cited = Classifier(input_size=feature_size, mode='cited')
clf = Classifier(input_size=feature_size, mode='total',
                 clf_feat=clf_feat, clf_cite=clf_cite, clf_cited=clf_cited)

# train
import torch.optim as optim
# P(C|Feat)
x_train_feature = torch.Tensor(x_test_feature)
optimizer = optim.SGD(clf_feat.parameters(), lr=0.01, momentum=0.5)



# TODO: test





