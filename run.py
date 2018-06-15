from model import Classifier
from utils import *


# path initialization
graph_path = 'Data/Cit-HepTh.txt'
train_path = 'Data/train.csv'
test_path = 'Data/test.csv'
info_path = 'Data/node_information'

# construct graph dataframe
graph_df = readGraph(graph_path)

# read node information
info_df = readInfo(info_path)

# read training set, make training graph
x_train_id, y_train_label = readIdLabel(train_path)
graph_train = graphTrain(graph_df, x_train_id)

# transform label to one-hot vector
y_train_onehot = label2onehot(y_train_label)

# read test set
x_test_id, _ = readIdLabel(test_path)

# read text
x_train_text = readText(info_df, x_train_id)
x_test_text = readText(info_df, x_test_id)

# extract features, make graph featured
x_train_feature, extractor = extractFeature(x_train_text)
x_test_feature = None # extractor.transform(x_test_text)
'''
TODO: concavate train and test features => feature_list
'''
feature_list = list()
# for training set
graph_train_weighted = graphWeighted(graph_train, x_train_feature)
graph_train_featured = graphFeatured(graph_train_weighted, x_train_feature)
# for whole data
graph_weigthed = graphWeighted(graph_df, feature_list)
graph_featured = graphFeatured(graph_weigthed, feature_list)

# make citation feature
# for training set
citing_train = None # citation condition matrix for training
cited_train = None # idem.
feature_train_citing = combineFeature(graph_train_featured, citing_train)
feature_train_cited = combineFeature(graph_train_featured, cited_train)
# for whole data
citing = None
cited = None
feature_citing = combineFeature(graph_train_featured, citing)
feature_cited = combineFeature(graph_train_featured, cited)
feature_test_citing = list() # feature_citing[test_id]
feature_test_cited = list() # feature_cited[test_id]

# initialization
feature_size = 10
clf_feat = Classifier(input_size=feature_size, mode='feature')
clf_citing = Classifier(input_size=feature_size, mode='citing')
clf_cited = Classifier(input_size=feature_size, mode='cited')
clf = Classifier(input_size=feature_size, mode='total',
                 clf_feat=clf_feat, clf_citing=clf_citing, clf_cited=clf_cited)

# TODO: train

# TODO: test





