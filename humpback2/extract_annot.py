import pandas as pd
import os

out = []
for fn in os.listdir('annots/'):
    a = open('annots/'+fn).read().split('<label ')[1:]
    for l in a:
        out.append({'fn':fn[:-3]+'aif', 'pos':l.split('"')[1], 'label':(l.split('"')[-2]).replace(' ','')})

df = pd.DataFrame().from_dict(out)
df.label = df.label.replace({'c':'C'})
grp = df.groupby('label').count()
df.drop(df[df.label.isin(grp[grp.fn<50].index)].index, inplace=True)
print(df.label.value_counts())
df.to_csv('humpback2.csv', index=False)
