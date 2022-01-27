# libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
 
# set width of bars
barWidth = 0.3
 
# set heights of bars
top1_raw = [89.25,95.00,94.64,92.37,87.67,82.92,83.77,90.58]
top5_raw = [98.44,99.57,99.32,99.22,98.00,96.09,96.55,98.85]
top1_b = [13.43,18.13,16.73,15.69,8.11,6.83,7.34,8.81]
top1_b_norm = [0.1504582756,0.1908877881,0.1767781202,0.1698722325,0.09250042574,0.08241447293,0.0875895978,0.09724190968]
top1_f = [81.69,89.19,85.77,86.55,75.04,68.03,71.70,78.39]
top1_f_norm = [0.915316312,0.9388833139,0.9063072597,0.9370710139,0.8558860055,0.82034928,0.8558937986,0.8653988704]
top5_b = [27.66,33.55,30.81,31.34,18.32,15.32,16.93,19.45]
top5_b_norm = [0.2810306035,0.336925698,0.3101730846,0.3159110281,0.1868934609,0.159408298,0.1753393031,0.1967635751]
top5_f = [92.44,95.77,93.25,94.89,87.89,83.44,86.50,89.69]
top5_f_norm = [0.9391290464,0.9618377956,0.9389224842,0.9563993247,0.8968961919,0.8683938992,0.8958294352,0.9073804189]

# Set position of bar on X axis
r1 = np.arange(len(top1_raw))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
plt.style.use('seaborn')
# create a color palette
sns.set_palette('colorblind')
palette = sns.color_palette('Paired')

# Make the plot
plt.bar(r1, top1_raw, width=barWidth, edgecolor='white', label='Full Image')
plt.bar(r2, top1_f, width=barWidth, edgecolor='white', label='Foreground')
plt.bar(r3, top1_b, width=barWidth, edgecolor='white', label='Background')
 
# Add xticks on the middle of the group bars
# plt.xlabel('model', size=22, **csfont)
plt.yticks(np.array((0, 20, 40, 60, 80, 100)), fontsize=22, **csfont)
plt.xticks([r + barWidth for r in range(len(top1_raw))], 
['ViT-B32','ViT-L16','ViT-L32','ViT-B16','ResNet50','MobileNetV2','DensNet121','ResNet152'],
rotation=45, size=18, **csfont)
plt.ylim(0, 115)
 
# Create legend & Show graphic
plt.legend(ncol=3, fontsize=30, frameon=True, facecolor="white", prop={'family':"Times New Roman", 'size': 16})

ax = plt.gca()
ax.xaxis.grid()


plt.ylabel("Top-1 accuracy (%)", size=22, **csfont)

plt.savefig("background-top1-acc.pdf", dpi=200, bbox_inches='tight')