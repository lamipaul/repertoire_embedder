import matplotlib.pyplot as plt
import pandas as pd, numpy as np

species = np.loadtxt('good_species.txt', dtype=str)
info = {
    'bengalese_finch1': ['bengalese finch', 'nicholson2017bengalese', 'bird'],
    'bengalese_finch2': ['bengalese finch', 'koumura2016birdsongrecognition', 'bird'],
    'california_thrashers': ['california trashers', 'arriaga2015bird', 'bird'],
    'cassin_vireo': ['cassin vireo', 'arriaga2015bird', 'bird'],
    'black-headed_grosbeaks': ['black-headed grosbeaks', 'arriaga2015bird', 'bird'],
    'zebra_finch': ['zebra finch', 'elie2018zebra', 'bird'],
    'humpback': ['humpback whale', 'malige2021use', 'cetacean'],
}

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
for i, specie in enumerate(species):
    df = pd.read_csv(f'{specie}/{specie}.csv')
    ax[i//3, i%3].bar(range(df.label.nunique() + 1), list(df.label.value_counts()) + [df.label.isna().sum()], log=True)
    ax[i//3, i%3].set_title(specie)
plt.tight_layout()
plt.savefig('annot_distrib.pdf')


a = "Specie & \# Classes & \# Samples & Annotations \% \\\\ \hline \n"
for specie in species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    a += f"{info[specie][0]} \cite{{{info[specie][1]}}} & {df.label.nunique()} & {len(df)} & {int(100*(~df.label.isna()).sum()/len(df))} \\\\ \hline \n"
f = open('annot_distrib.tex', 'w')
f.write(a)
f.close()
