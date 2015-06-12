from ReadData import *
import datetime
import numpy as np
from sklearn import svm
from sklearn import decomposition


startTime = datetime.datetime.now()
trainImage = getTrainImage()
trainLabel = getTrainLabel()
testImage = getTestImage()
testLabel = getTestLabel()

pca = decomposition.PCA(n_components=300, whiten=True).fit(trainImage)
trainImage_PCA = pca.transform(trainImage)
testImage_PCA = pca.transform(testImage)


# clf = svm.LinearSVC()
clf = svm.SVC(kernel = 'linear')
clf.fit(trainImage_PCA, trainLabel)


result = clf.predict(testImage_PCA)
errorCount = 0
for i in range(0, len(result)):
    if result[i] != testLabel[i]:
        errorCount += 1
endTime = datetime.datetime.now()
print('errorCount = ' + str(errorCount))
print('time elapse = ' + str(endTime - startTime))