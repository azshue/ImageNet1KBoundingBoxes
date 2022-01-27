# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns

font = {'family' : 'Times New Roman',
        'size'   : 20}

csfont = {'fontname':'Times New Roman'}

matplotlib.rc('font', **font)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=18)
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('figure', titlesize=22)
matplotlib.rcParams['font.family'] = ["Times New Roman"]
 
# Data
#low_pass
df=pd.DataFrame({'threshould': np.array((0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.08,0.05)), 
                'ViT-B/16': np.array((85.99,86.02,85.95,85.9,85.46,84.35,82.96,80.68,75.63,70.76,53.65)), 
                'ViT-L/16': np.array((87.08,87.09,87.06,86.94,86.8,86.42,85.2,83.3,79.62,75.81,62.91)),
                'ResNet-50': np.array((79.03,78.87,78.4,77,75.15,72.73,67.25,57.41,43.84,32.73,16.06)),
                'ResNet152d': np.array((83.69,83.64,83.42,82.59,81.45,79.54,76.04,69.53,58.72,51.8,28.76))})

low_pass_limits = [0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.08,0.05]
vit_b_low = [85.99,86.02,85.95,85.9,85.46,84.35,82.96,80.68,75.63,70.76,53.65]
vit_l_low = [87.08,87.09,87.06,86.94,86.8,86.42,85.2,83.3,79.62,75.81,62.91]
res50_low = [79.03,78.87,78.4,77,75.15,72.73,67.25,57.41,43.84,32.73,16.06]
res152d_low = [83.69,83.64,83.42,82.59,81.45,79.54,76.04,69.53,58.72,51.8,28.76]

high_pass_limits = [0.03,0.02,0.015,0.01,0.005,0.003,0.001,0]
vit_b_high = [11.27,25.77,39.454,54.832,68.744,68.744,75.76,85.99]
vit_l_high = [17.13,34.98,50.49,66.14,77.53,77.53,82.27,87.08]
res50_high = [10.78,23.1,31.76,40.588,49.11,59.49,59.49,79.03]
res152d_high = [24.12,43.48,57.41,63.24,73.51,78.23,78.22,83.68]

top5_high_pass_limits = [0.03,0.02,0.015,0.01,0.005,0.003,0]
top5_vit_b_high = [21.74,43.154,60.702,77.112,89.108,89.108,98.002]
top5_vit_l_high = [28.606,52.706,70.95,85.7,94.096,94.096,98.302]
top5_res50_high = [21.182,40.914,52.54,63.282,72.056,81.446,94.384]
top5_res152d_high = [40.622,65.29,79.146,84.344,91.274,94.154,96.738]

plt.style.use('seaborn')
# create a color palette
sns.set_palette('colorblind')
palette = sns.color_palette('Paired')


# multiple line plots
# plt.plot( 'threshould', 'ViT-B', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2)
plt.plot( top5_high_pass_limits, top5_vit_b_high, marker='o', markerfacecolor=palette[1], markersize=8, color=palette[0], linewidth=2, label='ViT-B/16')
plt.plot( top5_high_pass_limits, top5_vit_l_high, marker='o', markerfacecolor=palette[9], markersize=8, color=palette[8], linewidth=2, label='ViT-L/16')
plt.plot( top5_high_pass_limits, top5_res50_high, marker='o', markerfacecolor=palette[3], markersize=8, color=palette[2], linewidth=2, label='ResNet-50')
plt.plot( top5_high_pass_limits, top5_res152d_high, marker='o', markerfacecolor=palette[7], markersize=8, color=palette[6], linewidth=2, label='ResNet152-d')
plt.grid(b=True)
# ax = plt.gca()
# ax.invert_xaxis()

plt.ylim(0, 110)
plt.yticks(np.array((0, 20, 40, 60, 80, 100)), fontsize=22, **csfont)
# plt.xticks(np.array((0.5,0.4,0.3,0.2,0.1)), fontsize=18, **csfont)
plt.xticks(np.array((0, .005, .01, .015, .02, .025, .03)), fontsize=18, **csfont)

# show legend
plt.legend(loc='lower left',ncol=2, fontsize=30, frameon=True, facecolor="white", prop={'family':"Times New Roman", 'size': 16})

plt.xlabel("High-pass threshold", size=22, **csfont)
plt.ylabel("Top-5 accuracy on ImageNet (%)", size=22, **csfont)


# show graph
plt.savefig("freq_eval_high_pass_top5.png", dpi=200, bbox_inches='tight')