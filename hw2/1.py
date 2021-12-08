import numpy as np

print('Loading data ...')
data_root='./timit_11/'
test = np.load(data_root + 'test_11.npy')

import torch
from torch.utils.data import Dataset
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int64)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

BATCH_SIZE = 270
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        n = [4096, 2048, 1024, 512, 256, 128]
        n_mom = 0.7
        self.layer1 = nn.Linear(429, n[0])
        self.layer2 = nn.Linear(n[0], n[1])
        self.layer3 = nn.Linear(n[1], n[2])
        self.layer4 = nn.Linear(n[2], n[3])
        self.layer5 = nn.Linear(n[3], n[4])
        self.layer7 = nn.Linear(n[4], 429)
        self.bn1 = nn.BatchNorm1d(n[0], momentum=n_mom)
        self.bn2 = nn.BatchNorm1d(n[1], momentum=n_mom)
        self.bn3 = nn.BatchNorm1d(n[2], momentum=n_mom)
        self.bn4 = nn.BatchNorm1d(n[3], momentum=n_mom)
        self.bn5 = nn.BatchNorm1d(n[4], momentum=n_mom)
        self.bn6 = nn.BatchNorm1d(n[5], momentum=n_mom)
        self.ln1 = nn.LayerNorm(429)
        self.out = nn.Linear(4719, 39) 
        self.mdo1 = nn.Dropout(0.4)
        self.mdo2 = nn.Dropout(0.4)
        self.mdo3 = nn.Dropout(0.4)
        self.mdo4 = nn.Dropout(0.4)
        self.mdo5 = nn.Dropout(0.4)
        self.mdo6 = nn.Dropout(0.1)
        self.do = nn.Dropout(0.5)
        self.act_fn = nn.ReLU()
        self.act_fn2 = nn.Sigmoid()
        self.cv1 = nn.Conv1d(in_channels=39, out_channels=429, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.ln1(x)

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.mdo1(x)
        

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.mdo2(x)


        x = self.layer3(x)
        x = self.bn3(x)    
        x = self.act_fn(x)  
        x = self.mdo3(x)  

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.act_fn(x)
        x = self.mdo4(x)

        x = self.layer5(x)
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.mdo5(x)

        x = self.layer7(x)
        # print(x.shape)
        x = x.view(-1, 11, 39)
        x = x.permute(0, 2, 1)
        x = self.cv1(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.out(x)
        x = self.do(x)
        return x

test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load("./1.ckpt"))

"""Make prediction."""

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)


with open('1.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
