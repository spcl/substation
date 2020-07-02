import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    fig, axs = plt.subplots(ncols=4)

    data = pd.read_csv('perf_softmax.txt', sep=' ', header=None)
    data = data.rename(columns={0: "Func", 4: "Time"})
    sns.violinplot(data=data, x='Func', y='Time', ax=axs[0])

    data = pd.read_csv('perf_bdrln.txt', sep=' ', header=None)
    data = data.rename(columns={0: "Func", 7: "Time"})
    sns.violinplot(data=data, x='Func', y='Time', ax=axs[1])

    data = pd.read_csv('perf_drln.txt', sep=' ', header=None)
    data = data.rename(columns={0: "Func", 7: "Time"})
    sns.violinplot(data=data, x='Func', y='Time', ax=axs[2])

    data = pd.read_csv('perf_bad.txt', sep=' ', header=None)
    data = data.rename(columns={0: "Func", 5: "Time"})
    sns.violinplot(data=data, x='Func', y='Time', ax=axs[3])

    fig.set_size_inches(20,10)
    fig.savefig("violin.png")
