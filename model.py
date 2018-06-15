from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self, graph, graph_train, mode='train'):
        self.Graph = graph
        self.GraphTrain = graph_train
        self.Mode = mode
    

    def setMode(self, mode='train'):
        """
        set mode of classifier

        Args:
            mode: string type, 'train'/'test'
        
        Returns:
            none
        """
        self.Mode = mode
    

    def combineFeature(self, x, x_id):
        """
        combine features from cite/cited articles

        Args:
            x: sample feature
            x_id: sample ID
        
        Returns:
            feature combined from cite/cited articles
        """
        pass


    def fit(self, x, y):
        """
        fit classifier so that f(x) = y

        Args:
            x: combined list,
               x[0] features; x[1] IDs
            y: one-hot label
        
        Returns:
            None
        """
        pass


    def predict(self, x):
        """
        predict class from x

        Args:
            x: combined list,
               x[0] features; x[1] IDs
        
        Returns:
            probability of x belonging to different classes
        """
        pass
