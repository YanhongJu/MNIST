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



# clf = svm.SVC(kernel = 'rbf')
clf = svm.SVC(kernel = 'sigmoid')
# clf = svm.SVC()   gamma = 0.005 0.5 0 0.05
# clf = svm.SVC(kernel = 'linear',C=0.1)
clf.fit(trainImage, trainLabel)


result = clf.predict(testImage)
errorCount = 0
for i in range(0, len(result)):
    if result[i] != testLabel[i]:
        errorCount += 1
endTime = datetime.datetime.now()
print('errorCount = ' + str(errorCount))
print('time elapse = ' + str(endTime - startTime))