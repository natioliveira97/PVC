import numpy as np
import sys
import cv2 

cropping = False
 
def click_and_drop(event, x, y, flags, param):

	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.insert(0,(x, y))


################################## main #######################################
videoname = "data/" + sys.argv[1]+'.avi'
cap = cv2.VideoCapture(videoname)

filename = "data/" + sys.argv[1]+'_gab.txt'

file = open(filename,"w")

centers = []
i=0

while(True):
    ret, frame = cap.read()
    noball = [(-1,-1)]
    i=i+1
    refPt = []

    if ret==False:
    	break

    cv2.imshow('frame',frame)

    cv2.setMouseCallback('frame', click_and_drop) 

    option = cv2.waitKey(0)

    if option == ord('p'):
        centers.append(refPt)
        print(refPt)



    else:
        centers.append(noball)
        print(noball)

centers = np.array(centers)

shape = centers.shape

for i in range(shape[0]):
    line = str(centers[i][0][0]) + ' ' + str(centers[i][0][1]) + '\n'
    file.write(line)
        


file.close() 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
