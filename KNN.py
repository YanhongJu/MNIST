from sklearn import neighbors
from ReadData import *
import datetime
K=7;
startTime = datetime.datetime.now();
trainImage = getTrainImage()
trainLabel = getTrainLabel()
testImage = getTestImage()
testLabel = getTestLabel()
knn = neighbors.KNeighborsClassifier(algorithm = 'auto',leaf_size = 30,n_neighbors=K,warn_on_equidistant = True, weights = 'uniform')
print "length is " ,len(trainImage), " ", len(trainLabel)
knn.fit(trainImage,trainLabel)
match = 0;
for i in xrange(len(testLabel)):
    predictLabel = knn.predict(testImage[i])[0]
    print i,' '
    print predictLabel,' ', testLabel[i]
    if(predictLabel==testLabel[i]):
        match += 1

endTime = datetime.datetime.now()
print 'use time: '+str(endTime-startTime)
print 'error rate: '+ str(1-(match*1.0/len(testLabel)))