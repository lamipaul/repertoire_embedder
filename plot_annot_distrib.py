import matplotlib.pyplot as plt
import pandas as pd
import models

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 5))
for i, specie in enumerate(models.meta):
    df = pd.read_csv(f'{specie}/{specie}.csv')
    ax[i//3, i%3].bar(range(df.label.nunique() + 1), list(df.label.value_counts()) + [df.label.isna().sum()], log=True)
    ax[i//3, i%3].set_title(specie)
plt.tight_layout()
plt.savefig('annot_distrib.pdf')