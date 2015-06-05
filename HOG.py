#import matplotlib.pyplot as plt
import numpy
import time
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

startTime = int(round(time.time()) * 1000)
for pixels in trainImage:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	numpy.multiply(image, 255)	
	fd = hog(image, orientations=12, pixels_per_cell=(2, 2), 
					cells_per_block=(1, 1), visualise=False)
	X1.append(fd)

clf = svm.LinearSVC()
clf.fit(X1, Y1)
endTime = int(round(time.time()) * 1000)

for pixels in testImage:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	numpy.multiply(image, 255)	
	fd = hog(image, orientations=12, pixels_per_cell=(2, 2), 
					cells_per_block=(1, 1), visualise=False)
	X2.append(fd)

result = clf.predict(X2)

errorCount = 0
for i in range(0, len(result)):
	if result[i] != Y2[i]:
		errorCount += 1
print('errorCount = ' + str(errorCount))
print('time elapse = ' + str(endTime - startTime))
#dec = clf.decision_function([[1]])
#dec.shape[1]

#image = trainImage[0]
#print(len(trainImage[0]))
#image = color.rgb2gray(data.astronaut())
#print(len(image[0]), len(image))
#fd = hog(image, orientations=8, pixels_per_cell=(4, 4), 
#					cells_per_block=(1, 1), visualise=False)
#print(len(fd))



#print(hog_image[0])
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#
#ax1.axis('off')
#ax1.imshow(image, cmap=plt.cm.gray)
#ax1.set_title('Input image')
#
## Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
#plt.show()
#

