import networkx as nx
import nltk
import csv
import pandas as pd
import numpy as np


def readGraph(path):
    """
    read graph from .txt file

    Args:
        path: path of .txt file
        
    Returns: 
        a networks type graph
    """
    graph = nx.read_edgelist(path, delimiter='\t', create_using=nx.DiGraph())
    return graph


def readInfo(path):
    """
    read node information from .csv file

    Args:
        path: path of .csv file
    
    Returns:
        information dataframe
    """
    df = pd.read_csv(path)
    return df


def readIdLabel(path):
    """
    read id and label from .csv file

    Args:
        path: path of .csv file

    Returns: 
        id list, label list
    """
    id_list = list()
    label_list = list()
    with open(path, 'r') as f:
        next(f)
        for line in f:
            t = line.split(',')
            id_list.append(t[0])
            if t[1][:-1] != '':
                label_list.append(t[1][:-1])
    return id_list, label_list


def graphTrain(graph, id_list):
    """
    construct graph from training set

    Args:
        graph: networkx type graph
        id_list: IDs of training set
    
    Returns:
        graph constucted from training set
    """
    graph_train = graph.subgraph(id_list)
    return graph_train


def label2onehot(label_list):
    """
    transform class label list to one-hot vector list

    Args:
        label_list: label list

    Returns: 
        matrix type, each row contains a one-hot vector, shape MxC
    """
    unique = np.unique(label_list)
    label_matrix = np.zeros((len(label_list),unique.size))

    for idx, label in enumerate(label_list):
        label_matrix[idx][np.argwhere(unique == label)[0][0]] = 1
    return label_matrix


def readText(dataframe, id_list):
    """
    read text (abstract) from dataframe

    Args:
        dataframe: dataframe type, contains node information
        id_list: sample list containing IDs

    Returns: 
        text list
    """
    text_list = list()
    for id in id_list:
        text_list.append(dataframe.loc[dataframe['id'] == int(id)]['abstract'].iloc[0])
    return text_list


def extractFeature(text_list):
    """
    extract feature list from text list

    Args:
        text_list: text list

    Returns: 
        feature_list: matrix type, each row contains a feature vector, shape MxW
        extractor: extractor trained by training set
    """
    #feature extraction based on word frequency
    from sklearn.feature_extraction.text import TfidfVectorizer
    extractor = TfidfVectorizer(decode_error='ignore', stop_words='english')
    feature_list = extractor.fit_transform(text_list).toarray()
    return feature_list, extractor


def similarity(x, y):
    """
    calculate similarity between x and y

    Args:
        x: feature vector, shape 1xF
        y: feature vector, shape 1xF
        
    Returns:
        reel number indicating similarity
    """
    import torch.nn.functional as F
    import torch
    return F.cosine_similarity(torch.Tensor(x.reshape(1, -1)), 
                               torch.Tensor(y.reshape(1, -1)))
   

# to be discrepted
def graphWeighted(graph, feature_list):
    """
    reform graph with weights between each pair of nodes

    Args:
        graph: networkx type graph
        feature_list: feature list

    Returns: 
        graph reformed, each row contains edge weights according to similarity
    """
    graph_weighted = None
    return graph_weighted


# to be discrepted
def graphFeatured(graph_weighted, feature_list):
    """
    reform graph weighted so as to contain features

    Args:
        graph_weighted: graph weighted
        feature_list: feature list
    
    Returns:
        graph featured
    """
    graph_featured = None
    return graph_featured


def combineFeature(graph, id_list, feature_list):
    """
    combine feature of citing or cited articles

    Args:
        graph: original directed graph
        id_list: ID list
        feature_list: feature list
    
    Returns:
        feature combined, type list
    """
    feature_combined_list = list()
    for _, node in enumerate(graph.nodes()):
        feature = feature_list[id_list.index(node)]
        feature_combined = np.zeros(len(feature))
        for _, successor in enumerate(graph.successors(node)):
            feature_successor = feature_list[id_list.index(successor)]
            feature_combined += similarity(feature, feature_successor) * feature_successor
        feature_combined_list.append(feature_combined)
    return feature_combined_list


def evaluate(pred_proba, onehot):
    """
    evaluate prediction performance by cross-entropy

    Args:
        pred_proba: probability predicted for each sample belonging to different classes, shape NxC
        onehot: ground truth, shape NxC

    Returns: 
        1/N \sum{y_{ij} \log{p_{ij}}}
    """
    return np.mean(np.sum(np.log(pred_proba) * onehot, axis=1))
