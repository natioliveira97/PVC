# Lívia Gomes Costa Fonseca - 16/0034078
# Natalia Oliveira Borges - 16/0015863

import cv2
import numpy as np
import os
import sys


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def clustering_sift_array(folder_path, label_id, k):
	imgData = load_images_from_folder(folder_path)
	descriptor_list = []

	#Encontra descritores sift de todas as imagens e concatena em um array
	sift=cv2.xfeatures2d.SIFT_create()

	for i in range(len(imgData)):
		kp, des = sift.detectAndCompute(imgData[i],None)
		if(des.shape[0]>=k):
			descriptor_list.append(des)	

	descriptor_array = np.array(descriptor_list)
	descriptor_array = np.vstack(descriptor_list).astype(np.float32)


	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS

	compactness,labels,centers = cv2.kmeans(descriptor_array, k ,None, criteria, 10 ,flags)

	label = np.full((k,1),label_id)
	return centers, label


def clustering_sift_image(imgData, k):
	#Encontra descritores sift
	sift=cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(imgData,None)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS

	centers = None

	if(des.shape[0]>=k):
		compactness,labels,centers = cv2.kmeans(des, k ,None, criteria, 10 ,flags)

	return centers


def training_data_sift(k):
	folder_path = ["../data/train/Bedroom","../data/train/Coast","../data/train/Forest","../data/train/Highway",
					"../data/train/Industrial", "../data/train/InsideCity","../data/train/Kitchen","../data/train/LivingRoom",
					"../data/train/Mountain","../data/train/Office","../data/train/OpenCountry","../data/train/Store","../data/train/Street",
					"../data/train/Suburb","../data/train/TallBuilding"]

	train_centers = []
	train_labels = []

	for i in range(len(folder_path)):
		centers,labels = clustering_sift_array(folder_path[i],i,k)
		train_centers.append(centers)
		train_labels.append(labels)

	train_centers = np.array(train_centers)
	train_centers =  np.vstack(train_centers).astype(np.float32)
	train_labels = np.array(train_labels)
	train_labels =  np.vstack(train_labels).astype(np.float32)

	return train_centers, train_labels


def final_label(results):
	occurances = np.bincount(results)
	classification = np.argmax(occurances) 

	return classification	


def compute_acuracia(classification):
	col, row = classification.shape

	total = 0
	diagonal = 0

	for i in range(col):
		for j in range(row):
			total = total + classification[i][j]
			if i==j:
				diagonal = diagonal +classification[i][j]

	acuracia = (float(diagonal)/float(total))*100
	return(acuracia)


def compute_acuracia_class(classification):
	col, row = classification.shape

	mat = classification

	total = 0
	diagonal = 0

	for i in range(row):
		total = 0
		for j in range(col):
			total = total + classification[i][j]
			if i==j:
				diagonal = classification[i][j]
		for j in range(col):
			mat[i][j] = (classification[i][j]/total)*100

		acuracia = (float(diagonal)/float(total))*100
		print("Acuracia classe %s = %s%%" % (i, acuracia))


def classification_sift_knn(k):
	folder_path = ["../data/test/Bedroom","../data/test/Coast","../data/test/Forest","../data/test/Highway",
					"../data/test/Industrial", "../data/test/InsideCity","../data/test/Kitchen","../data/test/LivingRoom",
					"../data/test/Mountain","../data/test/Office","../data/test/OpenCountry","../data/test/Store","../data/test/Street",
					"../data/test/Suburb","../data/test/TallBuilding"]

	print("Treinando...\n")
	centers, labels = training_data_sift(k)
	knn = cv2.ml.KNearest_create()
	knn.train(centers, cv2.ml.ROW_SAMPLE, labels)
	
	print("Aplicando treinamento...\n")
	classification_hist = np.zeros((15,15))

	for j in range(len(folder_path)):
		testImages = load_images_from_folder(folder_path[j])
	
		for i in range(len(testImages)):
			sample = clustering_sift_image(testImages[i],k)
			if sample is not None:
				ret, results, neighbours, dist = knn.findNearest(sample, 1)
				results = results.reshape(k).astype(np.int64)	
				classification = final_label(results)
				classification_hist[j][classification] = classification_hist[j][classification]+1
			
			
	#print(classification_hist)
	acuracia = compute_acuracia(classification_hist)
	print("Acuracia = %s%%" % acuracia)
	compute_acuracia_class(classification_hist)
	

def classification_sift_svm(k):
	folder_path = ["../data/test/Bedroom","../data/test/Coast","../data/test/Forest","../data/test/Highway",
					"../data/test/Industrial", "../data/test/InsideCity","../data/test/Kitchen","../data/test/LivingRoom",
					"../data/test/Mountain","../data/test/Office","../data/test/OpenCountry","../data/test/Store","../data/test/Street",
					"../data/test/Suburb","../data/test/TallBuilding"]

	print("Treinando...\n")
	centers, labels = training_data_sift(k)

	labels = labels.reshape(labels.shape[0]).astype(np.int64)
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_POLY)
	svm.setDegree(2)
	svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))	
	svm.train(centers, cv2.ml.ROW_SAMPLE, labels)

	
	print("Aplicando treinamento ...\n")
	classification_hist = np.zeros((15,15))

	for j in range(len(folder_path)):
		testImages = load_images_from_folder(folder_path[j])
	
		for i in range(len(testImages)):
			sample = clustering_sift_image(testImages[i],k)
			if sample is not None:
				results = svm.predict(sample,True)
				results = results[1].reshape(k).astype(np.int64)
				classification = final_label(results)
				classification_hist[j][classification] = classification_hist[j][classification]+1
			
			
	#print(classification_hist)
	acuracia = compute_acuracia(classification_hist)
	print("Acuracia = %s%%" % acuracia)
	compute_acuracia_class(classification_hist)


def clustering_orb_array(folder_path, label_id, k):
	imgData = load_images_from_folder(folder_path)
	descriptor_list = []

	#Encontra descritores sift de todas as imagens e concatena em um array
	orb=cv2.ORB_create()
	
	for i in range(len(imgData)):
		kp = orb.detect(imgData[i],None)
		kp, des = orb.compute(imgData[i],kp)

		if des is not None:
			descriptor_list.append(des)	
	
	
	descriptor_array = np.array(descriptor_list)
	descriptor_array = np.vstack(descriptor_list).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers = cv2.kmeans(descriptor_array, k ,None, criteria, 10 ,flags)

	label = np.full((k,1),label_id)
	return centers, label
	

def clustering_orb_image(imgData, k):
	#Encontra descritores sift
	orb=cv2.ORB_create()
	kp = orb.detect(imgData,None)
	kp, des = orb.compute(imgData,kp)
	centers = None
	if des is not None:
		if des.shape[0]>=k:
			des = np.float32(des)
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			flags = cv2.KMEANS_RANDOM_CENTERS
			compactness,labels,centers = cv2.kmeans(des, k ,None, criteria, 10 ,flags)

	return centers


def training_data_orb(k):
	folder_path = ["../data/train/Bedroom","../data/train/Coast","../data/train/Forest","../data/train/Highway",
					"../data/train/Industrial", "../data/train/InsideCity","../data/train/Kitchen","../data/train/LivingRoom",
					"../data/train/Mountain","../data/train/Office","../data/train/OpenCountry","../data/train/Store","../data/train/Street",
					"../data/train/Suburb","../data/train/TallBuilding"]

	train_centers = []
	train_labels = []

	for i in range(len(folder_path)):
		centers,labels = clustering_orb_array(folder_path[i], i, k)
		
		train_centers.append(centers)
		train_labels.append(labels)
		
	
	train_centers = np.array(train_centers)
	train_centers =  np.vstack(train_centers).astype(np.float32)
	train_labels = np.array(train_labels)
	train_labels =  np.vstack(train_labels).astype(np.float32)

	return train_centers, train_labels
	

def classification_orb_knn(k):
	folder_path = ["../data/test/Bedroom","../data/test/Coast","../data/test/Forest","../data/test/Highway",
					"../data/test/Industrial", "../data/test/InsideCity","../data/test/Kitchen","../data/test/LivingRoom",
					"../data/test/Mountain","../data/test/Office","../data/test/OpenCountry","../data/test/Store","../data/test/Street",
					"../data/test/Suburb","../data/test/TallBuilding"]

	print("Treinando...\n")
	centers, labels = training_data_orb(k)
	knn = cv2.ml.KNearest_create()
	knn.train(centers, cv2.ml.ROW_SAMPLE, labels)
	
	classification_hist = np.zeros((15,15))

	print("Aplicando treinamento...\n")
	for j in range(len(folder_path)):

		testImages = load_images_from_folder(folder_path[j])
	
		for i in range(len(testImages)):
			sample = clustering_orb_image(testImages[i], k)
			if sample is not None:
				ret, results, neighbours, dist = knn.findNearest(sample, 1)
				results = results.reshape(k).astype(np.int64)	
				classification = final_label(results)
				classification_hist[j][classification] = classification_hist[j][classification]+1
				
			
	#print(classification_hist)
	acuracia = compute_acuracia(classification_hist)
	print("Acuracia = %s%%" % acuracia)

	compute_acuracia_class(classification_hist)

	
#main
if(sys.argv[1] == '--r1'):
	print("CLASSIFICADOR 1 (SIFT + KNN)\n")
	print("Numero de clusters:")
	k=int(input())
	classification_sift_knn(k)
elif(sys.argv[1] == '--r2'):
	print("CLASSIFICADOR 2 (SIFT + SVM)\n")
	print("Numero de clusters:")
	k=int(input())
	classification_sift_svm(k)
elif(sys.argv[1] == '--r3'):
	print("CLASSIFICADOR 3 (ORB + KNN)\n")
	print("Numero de clusters:")
	k=int(input())
	classification_orb_knn(k)






