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
    'otter': ['otter', '', ''],
    'humpback': ['humpback whale', 'malige2021use', 'cetacean'],
    'dolphin': ['bottlenose dolphin', 'sayigh2022sarasota', 'cetacean']
}

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
for i, specie in enumerate(species):
    df = pd.read_csv(f'{specie}/{specie}.csv')
    ax[i//4, i%4].bar(range(df.label.nunique() + 1), list(df.label.value_counts()) + [df.label.isna().sum()], log=True)
    ax[i//4, i%4].set_title(specie)
plt.tight_layout()
plt.savefig('annot_distrib.pdf')


a = "\\textbf{Specie and source} & \\textbf{\# Unit types} & \\textbf{\# Vocalisations} & \\textbf{\% Labelling} \\\\ \hline \n"
for specie in species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    a += f"{info[specie][0]} \cite{{{info[specie][1]}}} & {df.label.nunique()} & {len(df)} & {int(100*(~df.label.isna()).sum()/len(df))} \\\\ \hline \n"
f = open('annot_distrib.tex', 'w')
f.write(a)
f.close()
