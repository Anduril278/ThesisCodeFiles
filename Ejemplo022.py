from numpy import *
from sklearn import tree, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import *

import numpy as np
import itertools
import matplotlib.pyplot as plt

ClassNames = ['Normal', 'Asphyxia', 'Deaf']

vn = np.zeros(15)
vac = np.zeros(15)
vpr = np.zeros(15)
vpr1 = np.zeros(15)
vpr2 = np.zeros(15)
vpr3 = np.zeros(15)
vrc = np.zeros(15)
vrc1 = np.zeros(15)
vrc2 = np.zeros(15)
vrc3 = np.zeros(15)
vf1 = np.zeros(15)
vf11 = np.zeros(15)
vf12 = np.zeros(15)
vf13 = np.zeros(15)
for i in range(0, 15):
    vn[i] = 2*i+3
    filenameTrD = "P1TrD_%03d_001.txt" %(vn[i])
    filenameTrR = "P1TrR_%03d_001.txt" %(vn[i])

    filenameTsD = "P1TsD_%03d_001.txt" %(vn[i])
    filenameTsR = "P1TsR_%03d_001.txt" %(vn[i])

    dataD = np.loadtxt(filenameTrD, comments='#')
    dataR = np.loadtxt(filenameTrR, comments='#')

    dataDT = np.loadtxt(filenameTsD, comments='#')
    dataRT = np.loadtxt(filenameTsR, comments='#')

    n = len(dataR)
    nfct = len(dataD[0])
    nt = len(dataRT)

    clf1 = ExtraTreesClassifier(n_estimators=10,
                            max_depth=None,
                            min_samples_split=2, 
                            random_state=0)
    clf1 = clf1.fit(dataD, dataR)
    dataDP1 = clf1.predict(dataDT)
    dataDPp1 = clf1.predict_proba(dataDT)

    dataDPt1 = clf1.predict(dataD)
    PrMat1 = precision_score(dataR, dataDPt1, average='macro')
    Prt1 = precision_score(dataR, dataDPt1, average=None)
    Rct1 = recall_score(dataR, dataDPt1, average=None)
    F1t1 = f1_score(dataR, dataDPt1, average=None)

    AccN1 = accuracy_score(dataRT, dataDP1)
    Acc1 = accuracy_score(dataRT, dataDP1, normalize=False)
    PrMa1 = precision_score(dataRT, dataDP1, average='macro')
    PrMi1 = precision_score(dataRT, dataDP1, average='micro')
    Pr1 = precision_score(dataRT, dataDP1, average=None)
    RcMa1 = recall_score(dataRT, dataDP1, average='macro')
    RcMi1 = recall_score(dataRT, dataDP1, average='micro')
    Rc1 = recall_score(dataRT, dataDP1, average=None)
    F1Ma1 = f1_score(dataRT, dataDP1, average='macro') 
    F1Mi1 = f1_score(dataRT, dataDP1, average='micro') 
    F11 = f1_score(dataRT, dataDP1, average=None)
    
    vac[i] = AccN1
    vpr[i] = PrMa1
    vpr1[i] = Pr1[0]
    vpr2[i] = Pr1[1]
    vpr3[i] = Pr1[2]
    vrc[i] = RcMa1
    vrc1[i] = Rc1[0]
    vrc2[i] = Rc1[1]
    vrc3[i] = Rc1[2]
    vf1[i] = F1Ma1
    vf11[i] = F11[0]
    vf12[i] = F11[1]
    vf13[i] = F11[2]

    print '-------------------------------'
    print '-------------ET----------------'
    print filenameTrD
    print 'ACC: ', AccN1
    print 'Pr: ', PrMa1, PrMi1
    print Pr1
    print 'Rc: ', RcMa1, RcMi1
    print Rc1
    print 'F1: ', F1Ma1, F1Mi1
    print F11

print vn
print vac
print vpr
print vrc
print vf1

vnmax = max(vn)
vnmin = min(vn)
vacmax = max(vac)
vacmin = min(vac)
plt.figure()
plt.plot(vn, vac, 'bo-')
plt.xlabel('Window size')
plt.ylabel('Percentage')
plt.title('Accuracy classification score with respect to window size')
plt.axis([vnmin, vnmax, vacmin, vacmax])
plt.grid(True)
plt.savefig('AccuracyBGDT.pdf')
plt.savefig('AccuracyBGDT.jpg')
plt.savefig('AccuracyBGDT.eps')

vprmax = max(vpr)
vprmin = min(vpr)
vrcmax = max(vrc)
vrcmin = min(vrc)
vf1max = max(vf1)
vf1min = min(vf1)
vmax = max([vprmax, vrcmax, vf1max])
vmin = min([vprmin, vrcmin, vf1min])
plt.figure()
plt.plot(vn, vpr, 'bo-', vn, vrc, 'r*-', vn, vf1, 'g.-')
plt.xlabel('Window size')
plt.ylabel('Percentage')
plt.title('Classification Metrics with respect to window size')
plt.legend(['Precision score', 'Recall score', 'F-measure score'])
plt.axis([vnmin, vnmax, vmin, vmax])
plt.grid(True)
plt.savefig('MeatricsBGDT.pdf')
plt.savefig('MeatricsBGDT.jpg')
plt.savefig('MeatricsBGDT.eps')

vpr1min = min(vpr1)
vpr1max = max(vpr2)
plt.figure()
plt.plot(vn, vpr1, 'bo-', vn, vpr2, 'r*-', vn, vpr3, 'g.-')
plt.xlabel('Window size')
plt.ylabel('Percentage')
plt.title('Precision score with respect to window size')
plt.legend(ClassNames)
plt.axis([vnmin, vnmax, vpr1min, vpr1max])
plt.grid(True)
plt.savefig('PrecisionCBGDT.pdf')
plt.savefig('PrecisionCBGDT.jpg')
plt.savefig('PrecisionCBGDT.eps')

vrc1min = min(vrc1)
vrc1max = max(vrc2)
plt.figure()
plt.plot(vn, vrc1, 'bo-', vn, vrc2, 'r*-', vn, vrc3, 'g.-')
plt.xlabel('Window size')
plt.ylabel('Percentage')
plt.title('Recall score with respect to window size')
plt.legend(ClassNames)
plt.axis([vnmin, vnmax, vrc1min, vrc1max])
plt.grid(True)
plt.savefig('RecallCBGDT.pdf')
plt.savefig('RecallCBGDT.jpg')
plt.savefig('RecallCBGDT.eps')

vf11min = min(vf11)
vf11max = max(vf12)
plt.figure()
plt.plot(vn, vf11, 'bo-', vn, vf12, 'r*-', vn, vf13, 'g.-')
plt.xlabel('Window size')
plt.ylabel('Percentage')
plt.title('F-measure score with respect to window size')
plt.legend(ClassNames)
plt.axis([vnmin, vnmax, vf11min, vf11max])
plt.grid(True)
plt.savefig('F1CBGDT.pdf')
plt.savefig('F1CBGDT.jpg')
plt.savefig('F1CBGDT.eps')
plt.show()