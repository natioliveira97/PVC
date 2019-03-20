import numpy as np
import sys
import cv2 


def pixelCountCheck(blackMin, whiteMin, whiteMask, blackMask, pt1, pt2):

	height, width = whiteMask.shape
	canvas = whiteMask.copy()
	canvas[0:height, 0:width] = 0

	maskedWhite = canvas.copy()
	maskedBlack = canvas.copy()

	canvas = cv2.rectangle(canvas,pt1,pt2,255,-1)
	maskedWhite = canvas&whiteMask
	maskedBlack = canvas&blackMask

	whiteCount = cv2.countNonZero(maskedWhite)
	blackCount = cv2.countNonZero(maskedBlack)

	area = (pt1[0]-pt2[0])*(pt1[1]-pt2[1])*1.0

	if(whiteCount/area > whiteMin and blackCount/area > blackMin):
		return 1
	else:
		return 0

ball_cascade = cv2.CascadeClassifier('train/cascade.xml')

videoname = "data/"+sys.argv[1]+'.avi'
cap = cv2.VideoCapture(videoname)

param1 = 0.1
param2 = 0.5
valid = 0

nframes = 0
nframes_with_ball = 0
acertos = 0
falsepositive = 0

filename = "data/" + sys.argv[1]+'_gab.txt'

file = open(filename,"r")

centers = []

for line in file:
	data = line.split()
	x=data[0]
	y=data[1]
	center = [(x,y)]
	centers.append(center)

while(nframes<(len(centers)+1)):
	ret, frame = cap.read()

	if ret==False:
		break

	if int(centers[nframes][0][0])>0:
		nframes_with_ball = nframes_with_ball+1

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ball = ball_cascade.detectMultiScale(gray, 1.3, 5)

	whiteMask = cv2.inRange(gray, 100, 255)
	blackMask = cv2.inRange(gray, 0, 100)

	n=0
	count=False

	for (x,y,w,h) in ball:
		n=n+1
		valid =  pixelCountCheck(param1, param2, whiteMask, blackMask, (x,y),(x+w,y+h))
		if valid:
			x2=x+w
			y2=y+h
			if int(centers[nframes][0][0])>=x and int(centers[nframes][0][0])<=x2 and int(centers[nframes][0][1])>=y and int(centers[nframes][0][1])<=y2:
				#print("1 %s" % count)
				if(count == False):
					acertos = acertos+1
					count = True
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			if(falsepositive<n):
				falsepositive=n

		


	nframes = nframes+1

	cv2.imshow('ball_detection',frame)
	cv2.waitKey(0)

taxa_acertos = (float(acertos)/nframes_with_ball)*100
print("Taxa de acerto = %s %%" % taxa_acertos)
print("Numero maximo de falsos positivos = %s" % falsepositive)

cv2.destroyAllWindows()
file.close()