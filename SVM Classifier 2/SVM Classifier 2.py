import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from createFeature import *

# ------------------TRAINING------------------------------------

#Category A (Training)
ResultA = np.empty((0,15631), int)
LabelA = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Training\CatA_Train\*.png"):
	image_A = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_A)

	ResultA = np.append(ResultA,featureVector, axis=0)
	LabelA = np.append(LabelA,[['A']], axis=0)
print ('Category A done')


#Category B (Training)
ResultB = np.empty((0,15631), int)
LabelB = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Training\CatB_Train\*.png"):
	image_B = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_B)

	ResultB = np.append(ResultB,featureVector, axis=0)
	LabelB = np.append(LabelB,[['B']], axis=0)
print ('Category B done')	


#Category C (Training)
ResultC = np.empty((0,15631), int)
LabelC = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Training\CatC_Train\*.png"):
	image_C = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_C)

	ResultC = np.append(ResultC,featureVector, axis=0)
	LabelC = np.append(LabelC,[['C']], axis=0)
print ('Category C done')


#Category D (Training)
ResultD = np.empty((0,15631), int)
LabelD = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Training\CatD_Train\*.png"):
	image_D = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_D)

	ResultD = np.append(ResultD,featureVector, axis=0)
	LabelD = np.append(LabelD,[['D']], axis=0)
print ('Category D done')


#Category E (Training)
ResultE = np.empty((0,15631), int)
LabelE = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Training\CatE_Train\*.png"):
	image_E = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_E)

	ResultE = np.append(ResultE,featureVector, axis=0)
	LabelE = np.append(LabelE,[['E']], axis=0)
print ('Category E done')	


training_samples = np.vstack((ResultA, ResultB, ResultC, ResultD, ResultE))
training_labels = np.vstack((LabelA, LabelB, LabelC, LabelD, LabelE))


print (training_samples.shape)

# ------------------TESTING------------------------------------

#Category A 
TestA = np.empty((0,15631), int)
TestLabelA = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Testing\CatA_Test\*.png"):
	image_A = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_A)


	TestA = np.append(TestA,featureVector, axis=0)
	TestLabelA = np.append(TestLabelA,[['A']], axis=0)


#Category B
TestB = np.empty((0,15631), int)
TestLabelB = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Testing\CatB_Test\*.png"):
	image_B = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_B)

	TestB = np.append(TestB,featureVector, axis=0)
	TestLabelB = np.append(TestLabelB,[['B']], axis=0)


#Category C 
TestC = np.empty((0,15631), int)
TestLabelC = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Testing\CatC_Test\*.png"):
	image_C = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_C)

	TestC = np.append(TestC,featureVector, axis=0)
	TestLabelC = np.append(TestLabelC,[['C']], axis=0)


#Category D
TestD = np.empty((0,15631), int)
TestLabelD = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Testing\CatD_Test\*.png"):
	image_D = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_D)

	TestD = np.append(TestD,featureVector, axis=0)
	TestLabelD = np.append(TestLabelD,[['D']], axis=0)


#Category E
TestE = np.empty((0,15631), int)
TestLabelE = np.empty((0,1), int)
for img in glob.glob("C:\Users\User\Desktop\(Sharon) FYP\Swim Cat\Testing\CatE_Test\*.png"):
	image_E = cv2.imread(img)

	# Feature vector
	featureVector = createFeature4(image_E)

	TestE = np.append(TestE,featureVector, axis=0)
	TestLabelE = np.append(TestLabelE,[['E']], axis=0)


testing_samples = np.vstack((TestA, TestB, TestC, TestD, TestE))
testing_labels = np.vstack((TestLabelA, TestLabelB, TestLabelC, TestLabelD, TestLabelE))

no_of_samples = np.array([len(TestLabelA), len(TestLabelB), len(TestLabelC), len(TestLabelD), len(TestLabelE)])
print (no_of_samples)

print (testing_samples.shape)
print (testing_labels.shape)

# ------------------ SVM TRAINING --------------------------------
print ('SVM training starting')
from sklearn.svm import SVC
clf = SVC()
clf.fit(training_samples, training_labels.ravel())
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

print ('SVM training ends')

# ------------------------------------------------------------------

confusion_matrix = np.zeros([5,5])

for i,part_sample in enumerate(testing_samples):
	classified_label = clf.predict(part_sample.reshape(1,-1))
	classified_label = classified_label[0]
	#print (classified_label)

	actual_label = testing_labels[i]
	actual_label = actual_label[0]
	#print (actual_label)


	actual_label = str(actual_label)
	classified_label = str(classified_label)

	# if (classified_label is actual_label):
	# 	print ('Classification is correct')
	# else:
	# 	print ('Classification is wrong')

	# Changing to numbers
	if (actual_label is 'A'):
		AL = 0
	elif (actual_label is 'B'):
		AL = 1
	elif (actual_label is 'C'):
		AL = 2
	elif (actual_label is 'D'):
		AL = 3
	elif (actual_label is 'E'):
		AL = 4

	if (classified_label is 'A'):
		CL = 0
	elif (classified_label is 'B'):
		CL = 1
	elif (classified_label is 'C'):
		CL = 2
	elif (classified_label is 'D'):
		CL = 3
	elif (classified_label is 'E'):
		CL = 4

	# Populating the confusion matrix
	confusion_matrix[CL, AL] = confusion_matrix[CL, AL]+1

print (confusion_matrix)

# Normalizing it with number of elements
confusion_matrix[:,0] = confusion_matrix[:,0]/no_of_samples[0]
confusion_matrix[:,1] = confusion_matrix[:,1]/no_of_samples[1]
confusion_matrix[:,2] = confusion_matrix[:,2]/no_of_samples[2]
confusion_matrix[:,3] = confusion_matrix[:,3]/no_of_samples[3]
confusion_matrix[:,4] = confusion_matrix[:,4]/no_of_samples[4]


print (confusion_matrix)


average_accuracy = np.mean(confusion_matrix.diagonal())
print ('Average Accuracy = ',average_accuracy)

plt.figure(1)
plt.imshow(confusion_matrix, interpolation ='none', aspect = 'auto')
labels = ['Sky', 'Pattern', 'Dark', 'White','Veil']
plt.xticks(range(0,5), labels)
plt.yticks(range(0,5), labels)
plt.xlabel('Actual class')
plt.ylabel('Predicted class')
plt.show()
