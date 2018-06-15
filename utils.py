import networkx as nx
import nltk
import csv
import pandas as pd


def readGraph(path):
    """
    read graph from .txt file

    Args:
        path: path of .txt file
        
    Returns: 
        a networks type graph
    """
    df = nx.read_edgelist(path, delimiter='\t', create_using=nx.DiGraph())
    return df


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
    graph_train = graph
    return graph_train


def label2onehot(label_list):
    """
    transform class label list to one-hot vector list

    Args:
        label_list: label list

    Returns: 
        matrix type, each row contains a one-hot vector
    """
    return label_list


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
    return text_list


def extractFeature(text_list):
    """
    extract feature list from text list

    Args:
        text_list: text list

    Returns: 
        matrix type, each row contains a feature vector
    """
    feature_list = list()
    return feature_list
   

def graphWeighted(graph, feature_list, id_list):
    """
    reform graph with weights between each pair of nodes

    Args:
        graph: networkx type graph
        feature_list: feature list
        id_list: id list

    Returns: 
        graph reformed, each row contains edge weights according to similarity
    """
    return graph


def evaluate(pred_proba, onehot):
    """
    evaluate prediction performance by cross-entropy

    Args:
        pred_proba: probability predicted for each sample belonging to different classes
        onehot: ground truth

    Returns: 
        1/N \sum{y_{ij} \log{p_{ij}}}
    """
    perform = 0
    return perform
