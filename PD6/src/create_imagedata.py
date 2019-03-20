'''
Para rodar tem que rodar o programa com o nome do video como argumento (sem .avi)

Para fazer um positivo clique e arraste fazendo um triangulo em volta da bola e aperte em 'p'

Para fazer um negativo aperte 'n'

Para pular o frame aperte outra tecla
'''



import numpy as np
import sys
import cv2 

cropping = False

def click_and_drop(event, x, y, flags, param):

	global refPt, cropping, crop_img

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.insert(0,(x, y))
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.insert(1,(x, y))
		cropping = False
		crop_img = gray[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		cv2.imshow("cut", crop_img)

################################## main #######################################
videoname = 'data/'+sys.argv[1]+'.avi'
cap = cv2.VideoCapture(videoname)

file_positive = open("positive.info","a+")
file_negative = open("negative.txt","a+")

i = 0

while(True):
	ret, frame = cap.read()
	refPt=[]
	i=i+1

	if ret==False:
		break
	crop_img = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame',gray)

	cv2.setMouseCallback('frame', click_and_drop) 
	option = cv2.waitKey(0)

	if option == ord('p'):
		print("Positive")
		imagename = "data/positive/" + sys.argv[1]+'_'+str(i)+'.png'
		cv2.imwrite(imagename, crop_img)
		file_positive_line = imagename + ' ' + str(1) + ' ' + str(0) + ' ' + str(0) + ' ' + str(refPt[1][0]-refPt[0][0])+ ' ' + str(refPt[1][1]-refPt[0][1]) + '\n'
		file_positive.write(file_positive_line)
		cv2.destroyWindow("cut")
	elif option == ord('n'):
		print("Negative")
		imagename = "data/negative/" + sys.argv[1]+'_'+str(i)+'.png'
		cv2.imwrite(imagename, crop_img)
		file_negative_line = imagename + '\n'
		file_negative.write(file_negative_line)
		cv2.destroyWindow("cut")

file_positive.close()
file_negative.close()     
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()