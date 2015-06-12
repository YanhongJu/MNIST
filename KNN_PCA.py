from sklearn import neighbors
from ReadData import *
import datetime
import numpy as np
from sklearn import decomposition



def test(d):
	K=1;
	startTime = datetime.datetime.now()
	trainImage = getTrainImage()
	trainLabel = getTrainLabel()
	testImage = getTestImage()
	testLabel = getTestLabel()

	knn = neighbors.KNeighborsClassifier(algorithm = 'auto',leaf_size = 30,n_neighbors=K,warn_on_equidistant = True, weights = 'uniform')

	pca = decomposition.PCA(n_components=d, whiten=True).fit(trainImage)
	trainImage_PCA = pca.transform(trainImage)
	testImage_PCA = pca.transform(testImage)
	knn.fit(trainImage_PCA,trainLabel)

	match = 0;

	for i in xrange(len(testLabel)):
		predictLabel = knn.predict(testImage_PCA[i])
		if(predictLabel[0]==testLabel[i]):
			match += 1

	endTime = datetime.datetime.now()
	time = endTime-startTime
	# print 'use time: '+str(time)
	error_rate = 1-(match*1.0/len(testLabel))
	# print 'error rate: '+ str(error_rate)
	return error_rate, time