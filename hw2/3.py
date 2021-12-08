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
        n = 1268
        self.network = nn.Sequential(
            nn.Linear(429, n),
            nn.Linear(n,n),nn.BatchNorm1d(n),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(n,n),nn.BatchNorm1d(n),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(n,n),nn.BatchNorm1d(n),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(n,n),nn.BatchNorm1d(n),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(n,n),nn.BatchNorm1d(n),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(n, 39)
        )

    def forward(self, x):
        x = self.network(x)
        return x

test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load("./3-3.ckpt"))

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


with open('3.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
