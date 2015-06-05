import cv2
import numpy
import sklearn.cluster
from sklearn import svm
from ReadData import *

#img = cv2.imread('home.png')
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
testImage = getTestImage()
testLabel = list(getTestLabel())

clustercount = 100
X = []
sift = cv2.SIFT()
clf = svm.LinearSVC()
count = 0
for pixels in testImage[0 : 5000]:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	image = numpy.array(image, dtype=numpy.uint8)
	image = numpy.multiply(image, 100)
	kp, des = sift.detectAndCompute(image, None)
	if des != None:
		print(str(len(des)) + '\tlabel\t' + str(testLabel[count]))
		for feature in des:
			X.append(feature)
	count += 1

print('Using k-means to cluster')
km = sklearn.cluster.KMeans(clustercount)
km.fit(X[0 : 5000])
print('Obtaining projection for each training image')
X_SVM_TRAIN = []
Y_SVM_TRAIN = testLabel[0 : 5000]
noneCount = 0
for pixels in testImage[0 : 5000]:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	image = numpy.array(image, dtype=numpy.uint8)
	image = numpy.multiply(image, 255)
	kp, des = sift.detectAndCompute(image, None)
	projection = [0] * clustercount
	if des != None:
		result = km.predict(des)
		for cluster in result:
			projection[cluster] += 10
	else:
		noneCount += 1
		
	#print(projection)
	X_SVM_TRAIN.append(projection)
print('NoneCount' + str(noneCount))

print('Training SVM')
clf.fit(X_SVM_TRAIN, Y_SVM_TRAIN)
print('Testing SVM')
X_SVM_TEST = []
Y_SVM_TEST = testLabel[5000 : 9000]
for pixels in testImage[5000 : 9000]:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	image = numpy.array(image, dtype=numpy.uint8)
	image = numpy.multiply(image, 255)
	kp, des = sift.detectAndCompute(image, None)
	projection = [0] * clustercount
	if des != None:
		result = km.predict(des)
		for cluster in result:
			projection[cluster] += 10
	X_SVM_TEST.append(projection)

result = clf.predict(X_SVM_TEST)
errorCount = 0
for i in range(0, len(result)):
	if result[i] != Y_SVM_TEST[i]:
		errorCount += 1

print(errorCount)
#kp = sift.detect(image,None)
#img=cv2.drawKeypoints(gray,kp)
#kp, des = sift.detectAndCompute(image, None)

#cv2.imwrite('sift_keypoints.jpg',img)
