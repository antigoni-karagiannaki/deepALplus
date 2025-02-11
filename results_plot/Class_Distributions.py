# -----------------------------------
#  <STRATEGY> -- uc_merced_Imb
# -----------------------------------

# plot the class distribution of the dataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# class distribution in training set

# read csv file
df = pd.read_csv( "results_plot/uc_merced_imb_class_distr.csv", header=0, index_col=0)

labels = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
           'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
           'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']


# plot the class distribution
plt.figure(figsize=(12, 7))
sns.barplot(x=labels, y=df.iloc[:,1], color='royalblue', label='Random Sampling')
plt.legend()
plt.xticks(rotation=45, fontsize=7)
# show the class_dict.values() on the plt
for i, v in enumerate(df.iloc[:,1]):
    plt.text(i, v + 1, str(v), ha='center')

plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Set [uc_merced_imb]')
plt.show()
