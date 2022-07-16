import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("hyperparam_search.csv")
    print(data.columns)
    single_layer = data.loc[(data['Layer1_Size'] != 13) & (data['Layer2_Size'] == 0) & (data['Layer3_Size']==0) & (data['Learning_Rate'] == 0.075) & (data['Batch_Size']==64)]
    scatter = single_layer.plot.scatter(x="Layer1_Size", y="MSE_Loss", marker='x', s=100)
    scatter.set_ylabel("MSE Loss")
    scatter.set_xlabel("Neurons In Hidden Layer")
    scatter.set_xlim(0,110)
    scatter.set_xticks(range(0,120, 10))
    scatter.set_title("Standardised Mean Square Error Loss (MSE) vs\n Number of Neurons in Hidden Layer\n for a one layer network.\n Batch Size=64, Learning Rate=0.075")
    plt.grid()
    plt.subplots_adjust(top=0.8)
    plt.savefig("1-Layer.png")

    two_layer = data.loc[(data['Layer1_Size'] == 60) & (data['Layer2_Size'] != 0) & (data['Layer3_Size'] == 0) & (
                data['Learning_Rate'] == 0.075) & (data['Batch_Size'] == 64)]
    fig, ax = plt.subplots()
    scatter = two_layer.plot.scatter(x="Layer2_Size", y="MSE_Loss", marker='x', s=100, ax=ax)
    scatter.set_ylabel("MSE Loss")
    scatter.set_xlabel("Neurons In Second Hidden Layer")
    scatter.set_xlim(0, 110)
    scatter.set_xticks(range(0, 120, 10))
    scatter.set_title(
        "Standardised Mean Square Error Loss (MSE) vs\n Number of Neurons in Second Hidden Layer\n for a two layer network.\n First Hidden Layer Size=60 Neurons,\nBatch Size=64, Learning Rate=0.075")
    plt.grid()
    plt.subplots_adjust(top=0.75, bottom=0.2)
    x_line = range(0,120,10)
    line = [0.21093608438968658 for i in range(0,120,10)]
    ax.plot(x_line, line, 'r--',zorder=-1,label='Best Loss Found For a Single Layer Network')
    ax.legend(bbox_to_anchor=(0.9,-0.2))
    plt.savefig("2-Layer.png")
    plt.show()

    three_layer = data.loc[(data['Layer1_Size'] == 60) & (data['Layer2_Size'] == 40) & (data['Layer3_Size'] != 0) & (
            data['Learning_Rate'] == 0.075) & (data['Batch_Size'] == 64)]
    fig, ax = plt.subplots()
    scatter = three_layer.plot.scatter(x="Layer3_Size", y="MSE_Loss", marker='x', s=100, ax=ax)
    scatter.set_ylabel("MSE Loss")
    scatter.set_xlabel("Neurons In Second Hidden Layer")
    scatter.set_xlim(0, 110)
    scatter.set_xticks(range(0, 120, 10))
    scatter.set_title(
        "Standardised Mean Square Error Loss (MSE) vs\n Number of Neurons in Third Hidden Layer\n for a three layer network.\n First Hidden Layer Size=60 Neurons,\nSecond Hidden Layer Size=40 Neurons\nBatch Size=64, Learning Rate=0.075")
    plt.grid()
    plt.subplots_adjust(top=0.7, bottom=0.2)
    x_line = range(0, 120, 10)
    line = [0.20561860501766205 for i in range(0, 120, 10)]
    ax.plot(x_line, line, 'g--', zorder=-1, label='Best Loss Found For a Two Layer Network')
    ax.legend(bbox_to_anchor=(0.85, -0.2))
    plt.savefig("3-Layer.png")
    plt.show()