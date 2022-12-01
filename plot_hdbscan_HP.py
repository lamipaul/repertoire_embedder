import matplotlib.pyplot as plt
import pandas as pd, numpy as np


frontends = ['biosound', 'vggish', '256_logMel128', '256_logSTFT', '256_Mel128', '256_pcenMel128', '512_pcenMel128', '128_pcenMel128', '64_pcenMel128', '32_pcenMel128', '16_pcenMel128']

fn = open('hdbscan_HP.txt', 'r')

out = []
for l in fn.readlines():
    l = l[:-1].split(' ')
    if len(l)==2:
        specie, frontend = l[0], l[1]
    else:
        out.append({'specie':specie, 'frontend':frontend, 'nmi':float(l[0]), 'mcs':int(l[1]), 'ms': l[2], 'eps': float(l[3]), 'al': l[4]})
df = pd.DataFrame().from_dict(out)

df.ms = df.ms.replace('None', 0).astype(int)

df.to_csv('hdbscan_HP3.csv', index=False)

best = df.loc[df.groupby(["specie", 'frontend']).nmi.idxmax()]
res = [(conf, (grp.set_index(['specie', 'frontend']).nmi / best.set_index(['specie', 'frontend']).nmi).sum()) for conf, grp in df.groupby(['mcs', 'ms', 'eps', 'al'])]
conf = res[np.argmax([r[1] for r in res])]
print('best HP', conf)
conf = conf[0]

fig, ax = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True, sharey=True)
for s, grp in df.groupby('specie'):
    ax[0].scatter([grp[grp.frontend==f].nmi.max() for f in frontends], np.arange(len(frontends)))
ax[0].grid()
ax[1].grid()
ax[0].set_yticks(np.arange(len(frontends)))
ax[0].set_yticklabels(frontends)
for s, grp in df[(( df.mcs==conf[0] )&( df.ms==conf[1] )&( df.eps==conf[2] )&( df.al==conf[3] ))].groupby('specie'):
    ax[1].scatter([grp[grp.frontend==f].nmi.max() for f in frontends], np.arange(len(frontends)), label=s)
ax[1].legend(bbox_to_anchor=(1,1))
plt.tight_layout()
ax[0].set_title('Best HDBSCAN settings')
ax[1].set_title('Shared HDBSCAN settings')
ax[0].set_xlabel('NMI')
ax[1].set_xlabel('NMI')
plt.tight_layout()
plt.savefig('NMIs_hdbscan.pdf')
