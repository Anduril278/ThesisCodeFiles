"""
Created on Wed Jun 27 19:17:16 2018

@author: ErickSalvador
"""

# Load the Pima Indians diabetes dataset from CSV URL
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree, svm, metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
"""
######################################################  FUNCTIONS
"""

def count(dataR):
    unique_elements, counts_elements = np.unique(dataR, return_counts=True)
    return counts_elements

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""
###################################################### Dataset loading
"""
dataset = np.loadtxt('DATOS.csv', delimiter=",") 
proofdata = np.loadtxt('TESTDATABASE.csv', delimiter=",")

print(dataset.shape)
"""
###################################################### Separate target values from raw values
"""
dataD = dataset[:,0:101]
dataR = dataset[:,101]
                    
dataDT = proofdata[:,0:101]
dataRT = proofdata[:,101]

nt = len(dataRT)
ClassNames = ['Good conditions', 'Broken Bars', 'CC in A phase','CC in B phase','CC in C phase']
"""
###################################################### Decision Tree
"""

"""
###################################################### Bagging-> Decision Tree
"""
clf2 = BaggingClassifier(tree.DecisionTreeClassifier(),
                            max_samples=0.5, 
                            max_features=0.5)

clf2 = clf2.fit(dataD, dataR)
dataDP2 = clf2.predict(dataDT)
dataDPp2 = clf2.predict_proba(dataDT)

AccN2 = accuracy_score(dataRT, dataDP2)
Acc2 = accuracy_score(dataRT, dataDP2, normalize=False)
PrMa2 = precision_score(dataRT, dataDP2, average='macro')
PrMi2 = precision_score(dataRT, dataDP2, average='micro')
Pr2 = precision_score(dataRT, dataDP2, average=None)
RcMa2 = recall_score(dataRT, dataDP2, average='macro')
RcMi2 = recall_score(dataRT, dataDP2, average='micro')
Rc2 = recall_score(dataRT, dataDP2, average=None)
F1Ma2 = f1_score(dataRT, dataDP2, average='macro') 
F1Mi2 = f1_score(dataRT, dataDP2, average='micro') 
F12 = f1_score(dataRT, dataDP2, average=None)

print '-------------------------------'
print 'ACC: ', AccN2
print 'Pr: ', PrMa2, PrMi2
print Pr2
print 'Rc: ', RcMa2, RcMi2
print Rc2
print 'F1: ', F1Ma2, F1Mi2
print F12

"""
########################################################RandomForest

"""
clf3 = RandomForestClassifier(n_estimators=10)

clf3 = clf3.fit(dataD, dataR)
dataDP3 = clf3.predict(dataDT)
dataDPp3 = clf3.predict_proba(dataDT)

AccN3 = accuracy_score(dataRT, dataDP3)
Acc3 = accuracy_score(dataRT, dataDP3, normalize=False)
PrMa3 = precision_score(dataRT, dataDP3, average='macro')
PrMi3 = precision_score(dataRT, dataDP3, average='micro')
Pr3 = precision_score(dataRT, dataDP3, average=None)
RcMa3 = recall_score(dataRT, dataDP3, average='macro')
RcMi3 = recall_score(dataRT, dataDP3, average='micro')
Rc3 = recall_score(dataRT, dataDP3, average=None)
F1Ma3 = f1_score(dataRT, dataDP3, average='macro') 
F1Mi3 = f1_score(dataRT, dataDP3, average='micro') 
F13 = f1_score(dataRT, dataDP3, average=None)

print '-------------------------------'
print 'ACC: ', AccN3
print 'Pr: ', PrMa3, PrMi3
print Pr3
print 'Rc: ', RcMa3, RcMi3
print Rc3
print 'F1: ', F1Ma3, F1Mi3
print F13

"""
########################################################   ExtraTrees

"""
clf4 = ExtraTreesClassifier(n_estimators=10,
                            max_depth=None,
                            min_samples_split=2, 
                            random_state=0)

clf4 = clf4.fit(dataD, dataR)
dataDP4 = clf4.predict(dataDT)
dataDPp4 = clf4.predict_proba(dataDT)

AccN4 = accuracy_score(dataRT, dataDP4)
Acc4 = accuracy_score(dataRT, dataDP4, normalize=False)
PrMa4 = precision_score(dataRT, dataDP4, average='macro')
PrMi4 = precision_score(dataRT, dataDP4, average='micro')
Pr4 = precision_score(dataRT, dataDP4, average=None)
RcMa4 = recall_score(dataRT, dataDP4, average='macro')
RcMi4 = recall_score(dataRT, dataDP4, average='micro')
Rc4 = recall_score(dataRT, dataDP4, average=None)
F1Ma4 = f1_score(dataRT, dataDP4, average='macro') 
F1Mi4 = f1_score(dataRT, dataDP4, average='micro') 
F14 = f1_score(dataRT, dataDP4, average=None)

print '-------------------------------'
print 'ACC: ', AccN4
print 'Pr: ', PrMa4, PrMi4
print Pr4
print 'Rc: ', RcMa4, RcMi4
print Rc4
print 'F1: ', F1Ma4, F1Mi4
print F14

"""
###################################################### Weighted Average probabilities-> Pr macro
"""
clff1 = VotingClassifier(estimators=[ 
                                    ('bg', clf2), 
                                    ('RF', clf3),
                                    ('ET', clf4),
                                    ], 
                                    voting='soft', 
                                    weights=[PrMa2,PrMa3,PrMa4])

clff1 = clff1.fit(dataD, dataR)
dataDPf1 = clff1.predict(dataDT)
dataDPpf1 = clff1.predict_proba(dataDT)

AccNf1 = accuracy_score(dataRT, dataDPf1)
Accf1 = accuracy_score(dataRT, dataDPf1, normalize=False)
PrMaf1 = precision_score(dataRT, dataDPf1, average='macro')
PrMif1 = precision_score(dataRT, dataDPf1, average='micro')
Prf1 = precision_score(dataRT, dataDPf1, average=None)
RcMaf1 = recall_score(dataRT, dataDPf1, average='macro')
RcMif1 = recall_score(dataRT, dataDPf1, average='micro')
Rcf1 = recall_score(dataRT, dataDPf1, average=None)
F1Maf1 = f1_score(dataRT, dataDPf1, average='macro') 
F1Mif1 = f1_score(dataRT, dataDPf1, average='micro') 
F1f1 = f1_score(dataRT, dataDPf1, average=None)

print '-------------------------------'
print 'ACCproba: ', AccNf1
print 'Pr: ', PrMaf1, PrMif1
print Prf1
print 'Rc: ', RcMaf1, RcMif1
print Rcf1
print 'F1: ', F1Maf1, F1Mif1
print F1f1

## WeigthredAverageProbabilities -> Pr[0]
clff2 = VotingClassifier(estimators=[ 
                                    ('bg', clf2), 
                                    ('RF', clf3),
                                    ('ET', clf4),
                                    ], 
                                    voting='soft', 
                                    weights=[Pr2[0],Pr3[0],Pr4[0]])

clff2 = clff2.fit(dataD, dataR)
dataDPf2 = clff2.predict(dataDT)
dataDPpf2 = clff2.predict_proba(dataDT)

AccNf2 = accuracy_score(dataRT, dataDPf2)
Accf2 = accuracy_score(dataRT, dataDPf2, normalize=False)
PrMaf2 = precision_score(dataRT, dataDPf2, average='macro')
PrMif2 = precision_score(dataRT, dataDPf2, average='micro')
Prf2 = precision_score(dataRT, dataDPf2, average=None)
RcMaf2 = recall_score(dataRT, dataDPf2, average='macro')
RcMif2 = recall_score(dataRT, dataDPf2, average='micro')
Rcf2 = recall_score(dataRT, dataDPf2, average=None)
F1Maf2 = f1_score(dataRT, dataDPf2, average='macro') 
F1Mif2 = f1_score(dataRT, dataDPf2, average='micro') 
F1f2 = f1_score(dataRT, dataDPf2, average=None)

print '-------------------------------'
print 'ACC: ', AccNf2
print 'Pr: ', PrMaf2, PrMif2
print Prf2
print 'Rc: ', RcMaf2, RcMif2
print Rcf2
print 'F1: ', F1Maf2, F1Mif2
print F1f2

## WeigthredAverageProbabilities -> Rc
clff3 = VotingClassifier(estimators=[ 
                                    ('bg', clf2), 
                                    ('RF', clf3),
                                    ('ET', clf4),
                                    ], 
                                    voting='soft', 
                                    weights=[Rc2[0],Rc3[0],Rc4[0]])

clff3 = clff3.fit(dataD, dataR)
dataDPf3 = clff3.predict(dataDT)
dataDPpf3 = clff3.predict_proba(dataDT)

AccNf3 = accuracy_score(dataRT, dataDPf3)
Accf3 = accuracy_score(dataRT, dataDPf3, normalize=False)
PrMaf3 = precision_score(dataRT, dataDPf3, average='macro')
PrMif3 = precision_score(dataRT, dataDPf3, average='micro')
Prf3 = precision_score(dataRT, dataDPf3, average=None)
RcMaf3 = recall_score(dataRT, dataDPf3, average='macro')
RcMif3 = recall_score(dataRT, dataDPf3, average='micro')
Rcf3 = recall_score(dataRT, dataDPf3, average=None)
F1Maf3 = f1_score(dataRT, dataDPf3, average='macro') 
F1Mif3 = f1_score(dataRT, dataDPf3, average='micro') 
F1f3 = f1_score(dataRT, dataDPf3, average=None)

print '-------------------------------'
print 'ACC: ', AccNf3
print 'Pr: ', PrMaf3, PrMif3
print Prf3
print 'Rc: ', RcMaf3, RcMif3
print Rcf3
print 'F1: ', F1Maf3, F1Mif3
print F1f3

## WeigthredAverageProbabilities -> F1
clff4 = VotingClassifier(estimators=[ 
                                    ('bg', clf2), 
                                    ('RF', clf3),
                                    ('ET', clf4),
                                    ], 
                                    voting='soft', 
                                    weights=[F12[0],F13[0],F14[0]])

clff4 = clff4.fit(dataD, dataR)
dataDPf4 = clff4.predict(dataDT)
dataDPpf4 = clff4.predict_proba(dataDT)

AccNf4 = accuracy_score(dataRT, dataDPf4)
Accf4 = accuracy_score(dataRT, dataDPf4, normalize=False)
PrMaf4 = precision_score(dataRT, dataDPf4, average='macro')
PrMif4 = precision_score(dataRT, dataDPf4, average='micro')
Prf4 = precision_score(dataRT, dataDPf4, average=None)
RcMaf4 = recall_score(dataRT, dataDPf4, average='macro')
RcMif4 = recall_score(dataRT, dataDPf4, average='micro')
Rcf4 = recall_score(dataRT, dataDPf4, average=None)
F1Maf4 = f1_score(dataRT, dataDPf4, average='macro') 
F1Mif4 = f1_score(dataRT, dataDPf4, average='micro') 
F1f4 = f1_score(dataRT, dataDPf4, average=None)

print '-------------------------------'
print 'ACC: ', AccNf4
print 'Pr: ', PrMaf4, PrMif4
print Prf4
print 'Rc: ', RcMaf4, RcMif4
print Rcf4
print 'F1: ', F1Maf4, F1Mif4
print F1f4

## Selection
dataDP = dataDPf1
dataDPp = dataDPpf2
clf = clff2

print '-------------------------------'
Acc = [AccNf1, AccNf2, AccNf3, AccNf4]
Acc_i = Acc.index(max(Acc))
if Acc_i==0:
    dataDP = dataDPf1
    dataDPp = dataDPpf1
    clf = clff1
    print 
elif Acc_i==1:
    dataDP = dataDPf2
    dataDPp = dataDPpf2
    clf = clff2
elif Acc_i==2:
    dataDP = dataDPf3
    dataDPp = dataDPpf3
    clf = clff3
else:
    dataDP = dataDPf4
    dataDPp = dataDPpf4
    clf = clff4
print Acc
print Acc_i+1
print '-------------------------------'

"""
###################################################### COMPUTE CONFUSION MATRIdataD
"""
cnf_matrix = confusion_matrix(dataRT, dataDP)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=ClassNames,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=ClassNames, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
