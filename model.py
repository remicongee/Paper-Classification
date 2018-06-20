import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

class Classifier(nn.Module):
    def __init__(self, input_size=10, class_num = 28):
        super(Classifier, self).__init__()
        self.Linear = nn.Linear(input_size, class_num)


    def forward(self, x):
        """
        ouput prediction probability from feature x

        Args:
            x: if not 'total' mode: feature; 
               else: x[0] self feature, x[1] citing feature, x[2] cited feature
        
        Returns:
            probability of x belonging to different classes
        """
        output = F.relu(self.Linear(x))
        return output


class PaperSet(data.Dataset):
    def __init__(self, feature_list, label_list=None, train=True, label_dict=None):
        self.feature_list = feature_list
        self.label_list = list()
        self.train = train
        self.class_to_idx = label_dict
        if self.train:
            for label in label_list:
                self.label_list.append([self.class_to_idx[label]])
            assert len(feature_list) == len(label_list)

    def __getitem__(self, idx):
        if self.train:
            return torch.Tensor(self.feature_list[idx]), torch.LongTensor(self.label_list[idx])
        else:
            return torch.Tensor(self.feature_list[idx])

    def __len__(self):
        return len(self.feature_list)


def train(model, epoch, optimizer, features, labels, criterion, batch_size=32, label_dict=None):
    print("--------------Make training set--------------")
    dataset = PaperSet(feature_list=features, label_list=labels, train=True, label_dict=label_dict)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print("--------------finished--------------")
    print("--------------Start training--------------")
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(loader):
            # data, target = Variable(data), Variable(target)
            data, target = Variable(data), Variable(target.squeeze_().cuda())
            optimizer.zero_grad()
            output = model(data)
            # loss
            loss = criterion(output, target)
            loss.backward()
            # update
            optimizer.step()
            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data[0]))


def test(model, features, batch_size=32):
    print("--------------Make test set--------------")
    dataset = PaperSet(feature_list=features, train=False)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    print("--------------finished--------------")
    print("--------------Start testing--------------")
    model.eval()
    pred_list = list()
    for feature in loader:
        output = model(Variable(feature))
        output = F.softmax(output)
        pred_list.append(output.cpu().data.numpy())
    return pred_list

