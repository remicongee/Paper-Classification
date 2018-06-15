import networkx as nx
import nltk
import csv
import pandas as pd


def readGraph(path):
    """
    read graph from .txt file
    :param path: path of .txt file
    :return: a networks type graph
    """
    pass


def readIdLabel(path):
    """
    read id and label from .csv file
    :param path: path of .csv file
    :return: id list, label list
    """
    pass


def label2onehot(label_list):
    """
    transform class label list to one-hot vector list
    :param label_list: label list
    :return: matrix type, each row contains a one-hot vector
    """
    pass


def readText(dataframe):
    """
    read text (abstract) from dataframe
    :param dataframe: dataframe type, contains node information
    :return: text list
    """
    pass


def extractFeature(text_list):
    """
    extract feature list from text list
    :param text_list: text list
    :return: matrix type, each row contains a feature vector
    """
    pass

def graphWeighted(graph, feature_list):
    """
    reform graph with weights between each pair of nodes
    :param graph: networkx type graph
    :param feature_list: feature list
    :return: graph reformed, each row contains edge weights according to similarity
    """
    pass

def evaluate(pred_proba, onehot):
    """
    evaluate prediction performance by cross-entropy
    :param pred_proba: probability predicted for each sample belonging to different classes
    :param onehot: ground truth
    :return: 1/N \sum{y_{ij} \log{p_{ij}}}
    """
    pass
