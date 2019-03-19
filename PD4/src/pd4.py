import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D



def jacard(result, gt):
	intersection = result&gt
	union = result|gt
	n_intersection = cv2.countNonZero(intersection)
	n_union = cv2.countNonZero(union)
	jac = float(n_intersection)/float(n_union)

	return jac
#end jacard

def acuracia(result, gt):
	true_positive = result&gt
	n_true_positive = cv2.countNonZero(true_positive)

	height, width = result.shape

	result2 = result.copy()
	gt2 = gt.copy()

	result2 = cv2.inRange(result, 0, 2)
	gt2 = cv2.inRange(gt, 0, 2)				

	true_negative = result2&gt2
	n_true_negative = cv2.countNonZero(true_negative)
	
	acur = (float(n_true_positive)+float(n_true_negative))/float(height*width)
	return acur
#end acuracia

def segmentation_knn():
	img_size = 1000
	treining_size = 100

	print("METODO KNN\n")

	#Abre as imagens
	imgPath = "../data/images/vienna16.tif"
	img = cv2.imread(imgPath)
	if img is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath = "../data/gt/vienna16.tif"
	gt = cv2.imread(gtPath)
	if gt is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath2 = "../data/images/vienna5.tif"
	img2 = cv2.imread(imgPath2)
	if img2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath2 = "../data/gt/vienna5.tif"
	gt2 = cv2.imread(gtPath2)
	if gt2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath3 = "../data/images/vienna22.tif"
	img3 = cv2.imread(imgPath3)
	if img3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath3 = "../data/gt/vienna22.tif"
	gt3 = cv2.imread(gtPath3)
	if gt3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	#Redimensiona as imagens de treinamento para um tambanho menor
	img = cv2.resize(img, (treining_size, treining_size), cv2.INTER_AREA)
	gt = cv2.resize(gt, (treining_size, treining_size), cv2.INTER_AREA)
	gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de validacao
	img2 = cv2.resize(img2, (img_size, img_size), cv2.INTER_AREA)
	gt2 = cv2.resize(gt2, (img_size, img_size), cv2.INTER_AREA)
	gt2 = cv2.cvtColor(gt2, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de verificacao
	img3 = cv2.resize(img3, (img_size, img_size), cv2.INTER_AREA)
	gt3 = cv2.resize(gt3, (img_size, img_size), cv2.INTER_AREA)
	gt3 = cv2.cvtColor(gt3, cv2.COLOR_RGB2GRAY)


	height, width, depth = img.shape
	height2, width2, depth2 = img2.shape
	height3, width3, depth3 = img3.shape

	#Transforma a matriz de pixel em vetores de pixels 
	trainData = np.vstack(img).astype(np.float32)
	responses = np.hstack(gt).astype(np.float32)
	responses = responses.reshape(height*width,1)

	#Aplica o treinamento
	print("Treinando...")
	knn = cv2.ml.KNearest_create()
	knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

	#Transforma a imagem de teste em vetor 
	sample = np.vstack(img2).astype(np.float32)

	#Aplica o aprendizado na imagem
	print("Aplicando na imagem de validacao...")
	ret, results, neighbours, dist = knn.findNearest(sample, 1)

	#Transforma o vetor de resultado em matriz
	results = results.reshape(height2, width2).astype(np.uint8)
	results = cv2.inRange(results, 120, 255)


	jac = jacard(results, gt3)*100
	print("Jaccard %s%%" % jac)

	acur = acuracia(results, gt3)*100
	print("Acuracia %s%%\n" % acur)

	print("Pressione 'enter' para continuar.\n")

	#Melhora o resultado retirando outliers

	cv2.imshow("Resultado", results)
	cv2.imshow("Gt", gt2)
	cv2.imshow("Image", img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("../data/knn_validation_results.png", results)

	#Validacao e ajuste de resultados
	#Pega os pontos em que o algoritmo errou e faz mais treinamento
	results = cv2.resize(results, (treining_size, treining_size), cv2.INTER_AREA)
	gt2= cv2.resize(gt2, (treining_size, treining_size), cv2.INTER_AREA)

	test_results =  np.hstack(results).astype(np.float32)
	test_results = test_results.reshape(height*width,1)
	v_gt2 = np.hstack(gt2).astype(np.float32)
	v_gt2 = v_gt2.reshape(height*width,1)

	size, depth = test_results.shape
	new_traindata = []
	new_response = []

	for i in range(size):
		if(test_results[i] != v_gt2[i]):
			new_traindata.append(sample[i])
			new_response.append(v_gt2[i])
			
	
	new_traindata = np.array(new_traindata)
	new_response = np.array(new_response)
	new_traindata = np.concatenate((new_traindata, trainData), axis=0)
	new_response = np.concatenate((new_response,responses), axis=0)

	print("Refazendo treinamento com pixels errados...")


	#Aplica o treinamento
	knn = cv2.ml.KNearest_create()
	knn.train(new_traindata, cv2.ml.ROW_SAMPLE, new_response)

	#Transforma a imagem de avaliacao em vetor tambem
	sample = np.vstack(img3).astype(np.float32)

	#Aplica o aprendizado na imagem
	print("Aplicando na imagem de avaliacao")
	ret, results, neighbours, dist = knn.findNearest(sample, 3)

	#Transforma o vetor de resultado em matriz
	results = results.reshape(height3, width3).astype(np.uint8)
	results = cv2.inRange(results, 120, 255)

	jac = jacard(results, gt3)*100
	print("Jaccard %s%%" % jac)

	acur = acuracia(results, gt3)*100
	print("Acuracia %s%%" % acur)

	cv2.imshow("Resultado", results)
	cv2.imshow("Gt", gt3)
	cv2.imshow("Image", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("../data/knn_final_results.png", results)
#end segmentation_knn

def segmentation_svm():
	img_size = 1000
	treining_size = 400

	print("METODO SVM\n")

	#Abre as imagens
	imgPath = "../data/images/vienna16.tif"
	img = cv2.imread(imgPath)
	if img is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath = "../data/gt/vienna16.tif"
	gt = cv2.imread(gtPath)
	if gt is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath2 = "../data/images/vienna5.tif"
	img2 = cv2.imread(imgPath2)
	if img2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath2 = "../data/gt/vienna5.tif"
	gt2 = cv2.imread(gtPath2)
	if gt2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath3 = "../data/images/vienna22.tif"
	img3 = cv2.imread(imgPath3)
	if img3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath3 = "../data/gt/vienna22.tif"
	gt3 = cv2.imread(gtPath3)
	if gt3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	#Redimensiona as imagens de treinamento para um tambanho menor
	img = cv2.resize(img, (treining_size, treining_size), cv2.INTER_AREA)
	gt = cv2.resize(gt, (treining_size, treining_size), cv2.INTER_AREA)
	gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de validacao
	img2 = cv2.resize(img2, (img_size, img_size), cv2.INTER_AREA)
	gt2 = cv2.resize(gt2, (img_size, img_size), cv2.INTER_AREA)
	gt2 = cv2.cvtColor(gt2, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de verificacao
	img3 = cv2.resize(img3, (img_size, img_size), cv2.INTER_AREA)
	gt3 = cv2.resize(gt3, (img_size, img_size), cv2.INTER_AREA)
	gt3 = cv2.cvtColor(gt3, cv2.COLOR_RGB2GRAY)

	height, width, depth = img.shape
	height2, width2, depth2 = img2.shape
	height3, width3, depth3 = img3.shape

	#Transforma a matriz de pixel em vetores de pixels 
	traindata = np.vstack(img).astype(np.float32)
	responses = np.hstack(gt).astype(np.int64)
	responses = responses.reshape(height*width)
	sample = np.vstack(img2).astype(np.float32)

	print("Treinando...")
	

	# Treina svm 
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_POLY)
	svm.setDegree(5)
	svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))	
	svm.train(traindata, cv2.ml.ROW_SAMPLE, responses)

	print("Aplicando na imagem de validacao...")
	# Testa svm
	results = svm.predict(sample,True)
	results = results[1].reshape(height2, width2).astype(np.uint8)
	#results = cv2.inRange(results, 120, 255)


	cv2.imshow("Resultado svm", results)
	cv2.imshow("Gt", gt2)
	cv2.imshow("Image", img2)

	jac = jacard(results, gt2)*100
	print("Jacard %s%%" % jac)

	acur = acuracia(results, gt2)*100
	print("Acuracia %s%%\n" % acur)

	print("Pressione 'enter' para continuar.\n")

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("../data/svm_validation_results.png", results)


	#Validacao e ajuste de resultados
	#Pega os pontos em que o algoritmo errou e faz mais treinamento
	results = cv2.resize(results, (treining_size, treining_size), cv2.INTER_AREA)
	gt2= cv2.resize(gt2, (treining_size, treining_size), cv2.INTER_AREA)

	test_results =  np.hstack(results).astype(np.int64)
	test_results = test_results.reshape(treining_size*treining_size)
	v_gt2 = np.hstack(gt2).astype(np.int64)
	v_gt2 = v_gt2.reshape(treining_size*treining_size)


	new_traindata = []
	new_response = []

	for i in range(treining_size*treining_size):
		if(test_results[i] != v_gt2[i]):
			new_traindata.append(sample[i])
			new_response.append(v_gt2[i])



	new_traindata = np.array(new_traindata)
	new_response = np.array(new_response)
	new_traindata = np.concatenate((new_traindata, traindata), axis=0)
	new_response = np.concatenate((new_response,responses), axis=0)

	print("Refazendo treinamento com pontos errados...")

	# Treina svm 
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_POLY)
	svm.setDegree(5)
	svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001))	
	svm.train(new_traindata, cv2.ml.ROW_SAMPLE, new_response)

	#Transforma a imagem de avaliacao em vetor tambem
	sample = np.vstack(img3).astype(np.float32)

	# Testa svm
	results = svm.predict(sample,True)
	results = results[1].reshape(height3, width3).astype(np.uint8)
	results = cv2.inRange(results, 120, 255)

	jac = jacard(results, gt3)*100
	print("Jacard %s%%" % jac)

	acur = acuracia(results, gt3)*100
	print("Acuracia %s%%" % acur)

	cv2.imshow("Resultado", results)
	cv2.imshow("Gt", gt3)
	cv2.imshow("Image", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("../data/svm_final_results.png", results)
#end segmentation_svm


def segmentation_YCrCb2():

	#Abre as imagens
	imgPath = "../data/images/vienna16.tif"
	img = cv2.imread(imgPath)
	if img is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))	

	gtPath = "../data/gt/vienna16.tif"
	gt = cv2.imread(gtPath)
	if gt is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath2 = "../data/images/vienna5.tif"
	img2 = cv2.imread(imgPath2)
	if img2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath2 = "../data/gt/vienna5.tif"
	gt2 = cv2.imread(gtPath2)
	if gt2 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	imgPath3 = "../data/images/vienna22.tif"
	img3 = cv2.imread(imgPath3)
	if img3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(imgPath))

	gtPath3 = "../data/gt/vienna22.tif"
	gt3 = cv2.imread(gtPath3)
	if gt3 is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))


	#Redimensiona as imagens de treinamento para um tamanho menor
	img = cv2.resize(img, (1000, 1000), cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	gt = cv2.resize(gt, (1000, 1000), cv2.INTER_AREA)
	gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de teste
	img2 = cv2.resize(img2, (1000, 1000), cv2.INTER_AREA)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
	gt2 = cv2.resize(gt2, (1000, 1000), cv2.INTER_AREA)
	gt2 = cv2.cvtColor(gt2, cv2.COLOR_RGB2GRAY)

	#Redimensiona as imagens de teste
	img3 = cv2.resize(img3, (1000, 1000), cv2.INTER_AREA)
	img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2YCrCb)
	gt3 = cv2.resize(gt3, (1000, 1000), cv2.INTER_AREA)
	gt3 = cv2.cvtColor(gt3, cv2.COLOR_RGB2GRAY)


	jac = 0

	for Cr_min_it in range(25):
		for Cr_max_it in range(Cr_min_it, 25):
			for Cb_min_it in range(25):
				for Cb_max_it in range(Cb_min_it, 25):

					lowerBound = np.array([0, Cr_min_it*10, Cb_min_it*10])
					upperBound = np.array([255, Cr_max_it*10, Cb_max_it*10])

					mask = cv2.inRange(img, lowerBound, upperBound)

					jac_it = jacard(mask, gt)*100

					if jac <= jac_it:
						jac = jac_it
						Cr_min = Cr_min_it*10
						Cb_min = Cb_min_it*10
						Cr_max = Cr_max_it*10
						Cb_max = Cb_max_it*10

	#aplica na imagem de teste
	lowerBound = np.array([100, Cr_min, Cb_min])
	upperBound = np.array([255, Cr_max, Cb_max])

	results_int = cv2.inRange(img2, lowerBound, upperBound)	

	cv2.imshow("segmentacao", results_int)
	cv2.imshow("gt", gt2)

	jac = jacard(results_int, gt2)*100
	print("Jacard %s%%" % jac)

	acur = acuracia(results_int, gt2)*100
	print("acuracia %s%%" % acur)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	jac = 0

	for Cr_min_it in range(Cr_min-7, Cr_min+7):
		for Cr_max_it in range(Cr_max-7, Cr_max+7):
			for Cb_min_it in range(Cb_min-7, Cb_min+7):
				for Cb_max_it in range(Cb_max-7, Cb_max+7):

					lowerBound = np.array([100, Cr_min_it, Cb_min_it])
					upperBound = np.array([255, Cr_max_it, Cb_max_it])

					mask = cv2.inRange(img, lowerBound, upperBound)

					jac_it = jacard(mask, gt)*100

					if jac <= jac_it:
						jac = jac_it
						Cr_min2 = Cr_min_it
						Cb_min2 = Cb_min_it
						Cr_max2 = Cr_max_it
						Cb_max2 = Cb_max_it

	#aplica na imagem de teste
	lowerBound = np.array([100, Cr_min2, Cb_min2])
	upperBound = np.array([255, Cr_max2, Cb_max2])

	results_int = cv2.inRange(img3, lowerBound, upperBound)	

	cv2.imshow("segmentacao", results_int)
	cv2.imshow("gt", gt3)

	jac = jacard(results_int, gt3)*100
	print("Jacard %s%%" % jac)

	acur = acuracia(results_int, gt3)*100
	print("acuracia %s%%" % acur)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
#end segmentation_YCrCb2


def White_Black():

	
	#Abre o ground truth da imagem final
	gtPath = "../data/gt/vienna22.tif"
	gt = cv2.imread(gtPath)
	if gt is None:
	    raise FileNotFoundError("'{0}' could not be opened!".format(gtPath))

	gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

	result_white = gt.copy()
	result_black = gt.copy()

	height, width = gt.shape
	result_white[0:height, 0:width] = 255
	result_black[0:height, 0:width] = 0


	print("\nResultado com apenas predio")

	jac_white = jacard(result_white, gt)*100
	print("Jacard %s%%" % jac_white)

	acur_white = acuracia(result_white, gt)*100
	print("acuracia %s%%" % acur_white)


	print("\nResultado com apenas fundo")

	jac_black = jacard(result_black, gt)*100
	print("Jacard %s%%" % jac_black)

	acur_black = acuracia(result_black, gt)*100
	print("acuracia %s%%" % acur_black)


#main
if(sys.argv[1] == '--r1'):
	segmentation_knn()
elif(sys.argv[1] == '--r2'):
	segmentation_svm()
elif(sys.argv[1] == '--r3'):
	segmentation_YCrCb2()