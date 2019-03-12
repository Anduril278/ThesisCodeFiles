filenameTrD = 'P1TrD_031_001.txt'
filenameTrR = 'P1TrR_031_001.txt'

filenameTsD = 'P1TsD_031_001.txt'
filenameTsR = 'P1TsR_031_001.txt'

from numpy import *
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import *

import numpy as np
import itertools
import matplotlib.pyplot as plt

dataD = np.loadtxt(filenameTrD, comments='#')
dataR = np.loadtxt(filenameTrR, comments='#')

dataDT = np.loadtxt(filenameTsD, comments='#')
dataRT = np.loadtxt(filenameTsR, comments='#')

n = len(dataR)
nfct = len(dataD[0])
nt = len(dataRT)

#dataD = transpose(dataD)
#dataR = transpose(dataR)

#dataDT = transpose(dataDT)
#dataRT = transpose(dataRT)

#clf = tree.DecisionTreeClassifier()

#clf = BaggingClassifier(tree.DecisionTreeClassifier(),
#                            max_samples=0.5, 
#                            max_features=0.5)

#clf = RandomForestClassifier(n_estimators=10)

#clf = ExtraTreesClassifier(n_estimators=10,
#                            max_depth=None,
#                            min_samples_split=2, 
#                            random_state=0)

#clf =  svm.SVC(kernel='linear', C=0.01)

#clf = AdaBoostClassifier(n_estimators=100)

clf = GradientBoostingClassifier(n_estimators=100, 
                                learning_rate=1.0,
                                max_depth=1, 
                                random_state=0)

clf = clf.fit(dataD, dataR)
dataDP = clf.predict(dataDT)
dataDPp = clf.predict_proba(dataDT)

ClassNames = ['Normal', 'Asphyxia', 'Deaf']

AccN = accuracy_score(dataRT, dataDP)
Acc = accuracy_score(dataRT, dataDP, normalize=False)
PrMa = precision_score(dataRT, dataDP, average='macro')
PrMi = precision_score(dataRT, dataDP, average='micro')
Pr = precision_score(dataRT, dataDP, average=None)

print(AccN)
print(Acc, nt)
print(PrMa, PrMi)
print(Pr)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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

# Compute confusion matrix
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