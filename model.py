import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size=10, class_num = 28, mode='total', 
                 clf_feat=None, clf_citing=None, clf_cited=None):
        super(Classifier, self).__init__()
        self.Mode = mode
        self.ClfFeat = clf_feat
        self.ClfCiting = clf_citing
        self.ClfCited = clf_cited
        self.Linear = None
        if self.Mode != 'total':
            self.Linear = nn.Linear(input_size, class_num)
    

    def setMode(self, mode='feature'):
        """
        set mode of classifier

        Args:
            mode: string type, 'feature', 'citing', 'cited' and 'total'
        
        Returns:
            none
        """
        self.Mode = mode


    def forward(self, x):
        """
        ouput prediction probability from feature x

        Args:
            x: feature
        
        Returns:
            probability of x belonging to different classes
        """
        if self.Mode == 'total':
            output = self.ClfFeat(x) * self.ClfCiting(x) * self.ClfCited(x)
        elif self.Mode == 'feature' or self.Mode == 'citing' or self.Mode == 'cited':
            output = nn.ReLU(self.Linear(x))
            output = nn.Softmax(output)
        return output