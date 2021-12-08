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
BATCH_SIZE = 256
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.c1 = nn.Conv1d(in_channels=39, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(2048)
        self.do1 = nn.Dropout()
        self.c2 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(1024)
        self.do2 = nn.Dropout()
        self.c3 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(512)
        self.do3 = nn.Dropout()
        self.act_fn = nn.ReLU()
        self.l1 = nn.Linear(5632, 2048)
        self.lb1 = nn.BatchNorm1d(2048)
        self.ld1 = nn.Dropout()
        self.l2 = nn.Linear(2048, 512)
        self.lb2 = nn.BatchNorm1d(512)
        self.ld2 = nn.Dropout()
        self.out = nn.Linear(512, 39)
        self.ob = nn.BatchNorm1d(39)
        self.od = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, 11, 39)
        x = x.permute(0, 2, 1)

        x = self.c1(x)
        x = self.do1(x)
        x = self.act_fn(x)
        x = self.bn1(x)
        

        x = self.c2(x)
        x = self.do2(x)
        x = self.act_fn(x)
        x = self.bn2(x)

        x = self.c3(x)
        x = self.do3(x)
        x = self.act_fn(x)
        x = self.bn3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.l1(x)
        x = self.ld1(x)
        x = self.act_fn(x)
        x = self.lb1(x)

        x = self.l2(x)
        x = self.ld2(x)
        x = self.act_fn(x)
        x = self.lb2(x)

        x = self.out(x)
        x = self.ob(x)
        x = self.act_fn(x)
        x = self.od(x)
        
        return x


test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load("./2.ckpt"))

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


with open('2.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
