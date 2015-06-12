#import matplotlib.pyplot as plt
import numpy
from sklearn import neighbors
import datetime
from ReadData import *
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm

trainImage = getTrainImage()
trainLabel = list(getTrainLabel())
testImage = getTestImage()
testLabel = list(getTestLabel())

#X1 = trainImage[0 : 10000]
#X2 = trainImage [10000 : 20000]
X1 = []
X2 = []
Y1 = trainLabel
Y2 = testLabel 
K = 5

startTime = datetime.datetime.now()

for pixels in trainImage:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	numpy.multiply(image, 255)	
	fd = hog(image, orientations=12, pixels_per_cell=(4, 4), 
					cells_per_block=(3, 3), visualise=False)
	X1.append(fd)


for pixels in testImage:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	numpy.multiply(image, 255)	
	fd = hog(image, orientations=12, pixels_per_cell=(4, 4), 
					cells_per_block=(3, 3), visualise=False)
	X2.append(fd)

knn = neighbors.KNeighborsClassifier(algorithm = 'auto',leaf_size = 30,n_neighbors=K,warn_on_equidistant = True, weights = 'uniform')
print 'Dimension ' + str(len(X1[0]))
knn.fit(X1,Y1)
match = 0;

for i in xrange(len(Y2)):
	print 'i:' + str(i)
	predictLabel = knn.predict(X2[i])
	if(predictLabel[0]==Y2[i]):
		match += 1

endTime = datetime.datetime.now()
time = endTime-startTime
print 'use time: '+str(time)
error_rate = 1-(match*1.0/len(Y2))
print 'error rate: '+ str(error_rate)
# return error_rate, time

#dec = clf.decision_function([[1]])
#dec.shape[1]

#image = trainImage[0]
#print(len(trainImage[0]))
#image = color.rgb2gray(data.astronaut())
#print(len(image[0]), len(image))
#fd = hog(image, orientations=8, pixels_per_cell=(4, 4), 
#					cells_per_block=(1, 1), visualise=False)
#print(len(fd))
