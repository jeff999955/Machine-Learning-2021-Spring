import pandas as pd

data = pd.read_csv('prediction.csv')
n = len(data)
predict = []
for i in range(n):
    predict.append(data["Class"][i])
output = []

from itertools import groupby
iter = groupby(predict)
pwc = [x[0] for x in iter]
iter = groupby(predict)
pwcnt = [sum(1 for _ in group) for _, group in iter]
ls = [x for x in zip(pwc, pwcnt)]
new_predict = []
for i in range(len(ls)):
    item, num = ls[i]
    if num > 1 or i == 0 or i == len(ls) - 1: 
        new_predict += ([item] * num)
        continue
    new_predict.append(ls[i - 1][0])
# print(new_predict)

cnt = 0
for i in range(n):
    if predict[i] != new_predict[i]:
        cnt += 1
print(cnt)


with open('new_prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
