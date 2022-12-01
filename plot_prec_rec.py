import hdbscan
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

out = "\\textbf{Specie and source} & \\textbf{\# labels} & \\textbf{\# clusters} & \\textbf{\# discr. clusters} & \\textbf{\% clustered vocs} & \\textbf{\# missed labels} \\\\ \hline \n"

for specie in species:
    dic = np.load(f'{specie}/encodings//encodings_{specie}_256_pcenMel128_sparrow_encoder_decod2_BN_nomaxPool.npy', allow_pickle=True).item()
    idxs, X = dic['idxs'], dic['umap']
    df = pd.read_csv(f'{specie}/{specie}.csv')
    clusters = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, cluster_selection_epsilon=0.05, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
    df.loc[idxs, 'cluster'] = clusters.astype(int)
    mask = ~df.loc[idxs].label.isna()

    print(specie)
    labelled = df[~df.label.isna()]
    goodClusters, missedLabels = [], []
    for l, grp in labelled.groupby('label'):
        precisions = grp.groupby('cluster').fn.count() / labelled.groupby('cluster').fn.count()
        best = precisions.idxmax()
        goodClusters.extend(precisions[precisions > 0.9].index)
        if not (precisions > .9).any():
            missedLabels.append(l)
        # print(f'Best precision for {l} is for cluster {best} with {(df.cluster==best).sum()} points, \
        # with precision {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.cluster==best).sum():.2f}\
        # and recall {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.label==l).sum():.2f}')


    print(f'{len(goodClusters)} clusters would sort {df.cluster.isin(goodClusters).sum()/len(df)*100:.0f}% of samples')
    print(f'{len(goodClusters)/df.label.nunique():.1f} cluster per label in avg)')
    print(f'{len(missedLabels)} over {df.label.nunique()} missed labels')

    out += f"{info[specie][0]} \cite{{{info[specie][1]}}} & {df.label.nunique()} & {df.cluster.nunique()-1} & {len(goodClusters)} & {df.cluster.isin(goodClusters).sum()/len(df)*100:.0f} & {len(missedLabels)} \\\\ \hline \n"

f = open('cluster_distrib.tex', 'w')
f.write(out)
f.close()
