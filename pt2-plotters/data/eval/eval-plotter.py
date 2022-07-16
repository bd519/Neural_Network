import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    res = pd.read_csv("output_final_mod.csv")
    ax = sns.stripplot(x=res['Diff'], y = res["Type"], size=8,linewidth=0.001, marker="$\circ$", ec="face")
    plt.xlabel("Difference Between Predicted Price and Actual Price /$")
    plt.ylabel("Location Type")
    plt.xlim((-400000,400000))
    plt.xticks([-400000, -200000, 0 , 200000, 400000])
    plt.title("Difference Between Predicted Price and Actual Price /$\nFor Various Location Types")
    plt.subplots_adjust(left=0.27)
    plt.savefig("Location_Types.png")
    plt.show()

    # define top and bottom colormaps
    # This dictionary defines the colormap
    cdict = {'red': ((0.0, 0.0, 0.0),  # no red at 0
                     (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                     (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

             'green': ((0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
                       (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                       (1.0, 0.0, 0.0)),  # no green at 1

             'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                      (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0))  # no blue at 1
             }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

    res['Diff'] = abs(res['Diff'])
    ax = plt.scatter(x=res["longitude"], y=res["latitude"], c=res["Diff"], cmap=GnRd)
    plt.title("Scatter Plot Showing Location of Property\nWith Color Indicating Predictive Difference")
    plt.colorbar(label="Absolute Difference Between Predicted and Actual\nHouse Value /$")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("location_heat_map.png")
    plt.show()