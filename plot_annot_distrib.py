import matplotlib.pyplot as plt
import pandas as pd, numpy as np

species = np.loadtxt('good_species.txt', dtype=str)

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
for i, specie in enumerate(species):
    df = pd.read_csv(f'{specie}/{specie}.csv')
    ax[i//3, i%3].bar(range(df.label.nunique() + 1), list(df.label.value_counts()) + [df.label.isna().sum()], log=True)
    ax[i//3, i%3].set_title(specie)
plt.tight_layout()
plt.savefig('annot_distrib.pdf')


a = "Specie & \# Classes & \# Annotated samples & \# Samples & Proportion of annotations\\\\ \hline \n"
for specie in species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    a += f"{specie.replace('_',' ')} & {df.label.nunique()} & {(~df.label.isna()).sum()} & {len(df)} & {int(100*(~df.label.isna()).sum()/len(df))} \\\\ \hline \n"
f = open('annot_distrib.tex', 'w')
f.write(a)
f.close()
