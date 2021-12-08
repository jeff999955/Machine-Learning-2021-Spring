TRAIN_DATA_PATH = './covid.train.csv'
TEST_DATA_PATH = './covid.test.csv'
SAVE_PATH = './model.ckpt'
CSV_PATH = 'test.csv'

batch_size = 250
n_epoch = 5000
early = 400
lr = 1e-3
wd = 1e-3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import csv


def za_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDataSet(Dataset):
    def __init__(self, mode):
        self.mode = mode

        path = TRAIN_DATA_PATH if mode != 'test' else TEST_DATA_PATH
        with open(path, 'r') as f:
            data = pd.read_csv(path).drop(columns = ['id'])
            data = np.array(data).astype(float)
        
        features = [40, 41, 42, 43, 57, 57, 57, 57, 58, 59, 60, 61, 75, 75, 75, 75, 76, 77, 78, 79]
        if mode == 'test':
            data = data[:, features]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, features]
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 1]
            else:
                indices = [i for i in range(len(data)) if i % 10 == 1]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])
        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(mode):
    dataset = MyDataSet(mode)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last = False, pin_memory=True)
    return dataloader

class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        dim = [64, 32, 16, 4, 1]
        self.net = nn.Sequential(
                nn.Linear(input_dim, dim[0]),
                nn.ReLU(),
                nn.Linear(dim[0], dim[1]),
                nn.ReLU(),
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[3]),
                nn.ReLU(),
                nn.Linear(dim[3], dim[4]),
            )

        self.criterion = nn.MSELoss()
    def forward(self, x):
        return self.net(x).squeeze(1)
    def cal_loss(self, pred, target):
        loss = self.criterion(pred, target)
        l2_lambda = 1e-3
        l2_reg = torch.tensor(0.).to(device())
        for param in self.parameters():
            l2_reg += torch.norm(param,p=2)**2
        loss += l2_lambda * l2_reg *0.5
        return loss

def test(dataset, model):
    model.eval()
    prediction = []
    for x in dataset:
        x = x.to(device())
        with torch.no_grad():
            y = model(x)
            prediction.append(y.detach().cpu())
    prediction = torch.cat(prediction).numpy()
    prediction = np.maximum(np.zeros(prediction.shape), prediction)
    return prediction

def main():
    za_seed(42069)


    train_set, dev_set, test_set = prep_dataloader('train'), prep_dataloader('dev'), prep_dataloader('test')
    model = NN(train_set.dataset.dim).to(device())
    try:
        model.load_state_dict(torch.load(SAVE_PATH))
    except:
        pass
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0, amsgrad = True)

    min_loss = 1000

    for epoch in range(n_epoch):
        model.train()
        for x, y in train_set:
            optimizer.zero_grad()
            x, y = x.to(device()), y.to(device())
            prediction = model(x)
            loss = model.cal_loss(prediction, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        for x, y in dev_set:
            x, y = x.to(device()), y.to(device())
            with torch.no_grad():
                prediction = model(x)
                mse_loss = model.cal_loss(prediction, y)
            val_loss += mse_loss.detach().cpu().item() * len(x)
        val_loss /= len(dev_set.dataset)

        if val_loss < min_loss:
            min_loss = val_loss
            print(f"Saving model epoch: {epoch}, loss: {min_loss}")
            torch.save(model.state_dict(), SAVE_PATH)
            early = 400
        else:
            early -= 1

    ## test
    del model
    model = NN(train_set.dataset.dim).to(device())
    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location = 'cpu'))
    except:
        pass
    output = test(test_set, model)
    print('Saving result to', CSV_PATH)
    with open(CSV_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(output):
            writer.writerow([i, p])

if __name__ == "__main__":
    main()
