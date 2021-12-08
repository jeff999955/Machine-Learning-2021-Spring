import pandas as pd

m1 = pd.read_csv("1.csv")
m2 = pd.read_csv("2.csv")
m3 = pd.read_csv("3.csv")

n = len(m3["Class"])

output = [0] * n
for i in range(n):
    if m1["Class"][i] == m2["Class"][i]:
        output[i] = m1["Class"][i]
    else:
        output[i] = m3["Class"][i]

with open("prediction.csv", "w") as f:
    f.write("Id,Class\n");
    for i, c in enumerate(output):
        f.write(f"{i},{c}\n")
