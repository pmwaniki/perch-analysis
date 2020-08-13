from settings import perch_config,result_dir,manuscript_dir
from data.datasets import perch,chestray
from data.preprocess_image import image2array
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns



input_shape=(224,224,1)
labs2={1: 'C', 2: 'OI', 3: 'C & OI', 4: 'N', 5: 'U'}

perch_train=perch.train
perch_test=perch.test
chexray_train=chestray.train
chexray_test=chestray.test
perch=pd.concat([perch_train,perch_test])
perch['class']=(perch['labels']).map(perch_config.labels)


counts1=perch.groupby(["SITE",'class'],group_keys=False).apply(lambda df:pd.Series({'n':df.shape[0]})).reset_index()
counts1["N"]=counts1.groupby('SITE')['n'].transform('sum')

counts2=counts1.groupby("class")['n'].agg('sum').reset_index()
counts2['N']=counts2['n'].sum()
counts2['SITE']="Total"
counts3=pd.concat([counts1,counts2],axis=0,sort=False)
counts3['perc']=counts3['n']/counts3['N']*100
counts3['summary']=counts3.apply(lambda row:"%d(%.1f)" % (row['n'],row['perc']),axis=1)

counts=pd.crosstab(counts3['SITE'],counts3['class'],values=counts3['summary'],aggfunc=lambda x:x)
counts=counts.loc[list(perch['SITE'].unique())+["Total",],perch_config.labels.values()]
counts.columns=counts.columns.map({'Consolidation': 'Consolidation',
                                                    'Other Infiltrate': 'Other Infiltrate',
       'Consolidation and Other Infiltrate':'\\multicolumn{1}{c}{Consolidation and\\\\ Other Infiltrate}', 'Normal':'Normal',
                                   'Uninterpretable':'Uninterpretable'})
numbers=pd.crosstab(perch['class'],perch.SITE,margins=True,margins_name="Total",normalize='columns')

with open(os.path.join(manuscript_dir, "classification_by_site.tex"), 'w') as f:
    counts.to_latex(f,escape=False)

#bar plot
p_counts=counts1
p_counts['perc']=p_counts['n']/p_counts["N"]*100
sites_=p_counts['SITE'].unique()
cats_=p_counts['class'].unique()

tot_height=np.zeros(len(sites_))
for i,c in enumerate(cats_):
    bar_heights=np.array([p_counts.loc[(p_counts["SITE"]==s) & (p_counts['class']==c),"n"].values[0] for s in sites_])
    bar_perc = np.array(
        [p_counts.loc[(p_counts["SITE"] == s) & (p_counts['class'] == c), "perc"].values[0] for s in sites_])
    plt.bar(sites_,
            bar_heights,
            label=c,bottom=tot_height)
    for k,s in enumerate(sites_):
        plt.text(s,tot_height[k],"%d(%.0f)" % (bar_heights[k],bar_perc[k]),
                 horizontalalignment='center',fontsize=7)
    tot_height=tot_height+bar_heights
plt.legend(loc=2,fontsize="small")
plt.xlabel("Site")
plt.ylabel("Number of CXRs")
plt.savefig(os.path.join(manuscript_dir, "classification_by_site.png"),dpi=300)
plt.show()
#numbers.to_csv(os.path.join(manuscript_dir, "classification by site.csv"))

## AGREEMENT BETWEEN RATERS

concodance=pd.melt(perch,id_vars=['SITE', 'CXRIMGID','FINALCONC'],
                   value_vars=['REV1', 'REV2','ARB1', 'ARB2',],
                   var_name="rater",value_name="rating")
concodance=concodance[~np.isnan(concodance['rating'])]
concodance['rater']=concodance.rater.map({'REV1':"Reviewer 1", 'REV2': "Reviewer 2", 'ARB1':"Arbitrator 1", 'ARB2':"Arbitrator 2"})
concodance['correct']=concodance['rating']==concodance['FINALCONC']
# concodance['class_rater']=(concodance['rating']-1).map(perch_config.labels)
# concodance['class_final']=(concodance['FINALCONC']-1).map(perch_config.labels)

concodance2=concodance.groupby(['rater','SITE'])['correct'].mean()
concodance3=concodance2.unstack()


with open(os.path.join(manuscript_dir, "accuracy_by_site_rater.tex"), 'w') as f:
    (concodance3*100).to_latex(f,float_format="%.0f")

fig,axs=plt.subplots(len(np.unique(concodance['SITE'])),
                     len(np.unique(concodance['rater'])),
                     gridspec_kw = {'wspace':0, 'hspace':0},
                     sharex='col',sharey='row',figsize=(12,15))
for i,s in enumerate(np.unique(concodance['SITE'])):
    for j,r in enumerate(np.unique(concodance['rater'])):
        # s = 'KEN'
        # r = 'ARB2'
        sub = concodance.loc[(concodance.rater == r) & (concodance.SITE == s), :]
        sub= sub[~np.isnan(sub['rating'])]
        cm = confusion_matrix(sub['FINALCONC'], sub['rating'], labels=list(labs2.keys()))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        axs[i][j].matshow(cm,cmap=plt.cm.Blues)
        axs[i][j].set_xticklabels(['']+list(labs2.values()),rotation=90,va='bottom')
        axs[i][j].set_yticklabels(['']+list(labs2.values()))
        if j==0:axs[i][j].set_ylabel(s)
        axs[i][j].set_xlabel(r)

# for a in axs.flat:
#     a.label_outer()
# axs.set_xlabel("Final")
fig.show()


#PLOT OF PERCH IMAGES
np.random.seed(123)
random_images=perch.groupby(['class','SITE'])['path'].agg(lambda x:np.random.choice(x,size=1))

fig,ax=plt.subplots(ncols=len(perch['SITE'].unique()),
                    nrows=len(perch['class'].unique()),
                    figsize=(12,7),sharex=True,sharey=True,
                    gridspec_kw = {'wspace':0, 'hspace':0})

for i,c in enumerate(perch['class'].unique()):
    for j,s in enumerate(perch['SITE'].unique()):
        file_path=random_images.loc[c][s]
        image=image2array(file_path,input_shape)
        ax[i][j].imshow(image[:,:,0],cmap="gray",vmin=0,vmax=1)
        # ax[i][j].axis("off")
        ax[i][j].set_xlabel(s)
        ax[i][j].set_ylabel(c.replace(" ","\n"))
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        # ax[i][j].set_aspect("equal")

for a in ax.flat:
    a.label_outer()
# fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(os.path.join(result_dir,"descriptives/perch_images.jpeg"))
plt.show()

#PLOT RANDOM CHESTRAY IMAGES
np.random.seed(452)
random_chestray=np.random.choice(chestray.train['path'], size=20)
fig,ax=plt.subplots(ncols=5,
                    nrows=4,
                    figsize=(12,7),sharex=True,sharey=True,
                    gridspec_kw = {'wspace':0, 'hspace':0})
for i,a in enumerate(ax.flat):
    image=image2array(random_chestray[i],input_shape)
    a.imshow(image[:,:,0],cmap="gray",vmin=0,vmax=1)
    a.axis('off')
plt.savefig(os.path.join(result_dir,"descriptives/chestray_images.jpeg"))
plt.show()

# PLOT OF RANDOM PERCH AND CHESTRAY IMAGES
np.random.seed(12)
random_chestray=np.random.choice(chestray.train['path'], size=4)
random_perch=np.random.choice(perch['path'], size=4)
random_chestray_perch=np.concatenate([random_chestray,random_perch])

fig,ax=plt.subplots(ncols=4,
                    nrows=2,
                    figsize=(9,5),sharex=True,sharey='row',
                    gridspec_kw = {'wspace':0, 'hspace':0})
for i,a in enumerate(ax.flat):
    image=image2array(random_chestray_perch[i],input_shape)
    a.imshow(image[:,:,0],cmap="gray",vmin=0,vmax=1)
    a.set_xticks([])
    a.set_yticks([])
    a.set_ylabel("Chestray-14" if i<4 else "PERCH")
    a.label_outer()
plt.savefig(os.path.join(manuscript_dir,"chestray_perch_images.jpeg"))
plt.show()




perch['_AGEM'].hist()
plt.xlabel("Age in months")
plt.show()

confusion_matrix(perch['REV1'],perch['REV2'])
confusion_matrix(perch['FINALCONC'],perch['REV1'])
plt.matshow(confusion_matrix(perch['FINALCONC'],perch['REV2']))
plt.colorbar()
plt.show()


accuracy_score(perch['REV2'],perch['REV1'])
accuracy_score(perch['FINALCONC'],perch['REV1'])
accuracy_score(perch['FINALCONC'],perch['REV2'])


reviewers=['REV1', 'REV2', 'ARB1', 'ARB2', 'FINALCONC']


def format_labels(x):
    print('Yo dawg, I got called')
    return 'perch_config.labels[x]'

fig,ax=plt.subplots(4,4,sharex=True,sharey=True,figsize=(15,15))
for i in range(5):
    for j in range(0,i):
        tab=perch[[reviewers[i],reviewers[j]]].copy()
        tab.dropna("index",inplace=True)
        c_m=confusion_matrix(tab[reviewers[i]],tab[reviewers[j]])
        c_m=c_m/c_m.max()
        ax[i-1,j].matshow(c_m)
        #ax[i - 1, j].set_yticks(ticks=[0,1,2,3,4])
        #ax[i-1,j].set_xticklabels(labels=[perch_config.labels[i] for i in range(5)],rotation = 45)

        if j==0: ax[i-1,j].set_ylabel(reviewers[i])
        if i==4: ax[i-1, j].set_xlabel(reviewers[j])
        ax[i - 1, j].format_ydata = format_labels
    for j in range(i+1,4):
        fig.delaxes(ax[i,j])

plt.show()


#Perch agreement by patient's age
perch['agreement']=perch['REV1']==perch['REV2']
perch['agecat']=perch['_AGEM'].map(lambda x: "(0-12)" if x<12 else "[12-60)")

from statsmodels.nonparametric.smoothers_lowess import lowess
y2=lowess(perch['agreement']*1.0,perch['_AGEM'],)



plt.plot(y2[:,0],y2[:,1])
plt.xlabel("Age in months")
plt.ylabel("Reader agreement (%)")
plt.show()

#reader agreement by site
perch.groupby("SITE").agg({'agreement':np.mean})
perch.groupby("agecat").agg({'agreement':np.mean})
perch.groupby("labels").agg({'agreement':np.mean})

for s in perch['SITE'].unique():
    subset = perch[perch['SITE'] == s]

    # Draw the density plot
    sns.distplot(subset['_AGEM'], hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label=s)

# Plot formatting
plt.legend(prop={'size': 16}, title='Site')
plt.xlabel('Age (months)')
plt.ylabel('Density')
plt.show()

sns.violinplot(x="SITE",y="_AGEM",data=perch)
plt.xlabel('Site')
plt.ylabel('Age in months')
plt.show()