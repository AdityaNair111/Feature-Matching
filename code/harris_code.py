import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_interest_points(image, feature_width):
	# Harris Corner Detector
	filter = cv2.getGaussianKernel(ksize=5,sigma=5) #initally filtering with a gaussian kernel
	filter = np.dot(filter,filter.transpose())
	image  = cv2.filter2D(image,cv2.CV_32F,filter)
	ix  = cv2.Sobel(image,cv2.CV_32F,1,0,1)
	iy  = cv2.Sobel(image,cv2.CV_32F,0,1,1)
	ix2 = ix * ix
	iy2 = iy * iy
	ixy = ix * iy
	
	sx2 = cv2.filter2D(ix2,cv2.CV_32F,filter)
	sy2 = cv2.filter2D(iy2,cv2.CV_32F,filter)
	sxy = cv2.filter2D(ixy,cv2.CV_32F,filter)
	lamda = 0.04
	r = ((sx2 * sy2) - (sxy * sxy)) -  lamda*((sx2 + sy2)**2) # Det(H) - lambda*(Trace(H))^2 (response function)

	# Adaptive Non-Maximal Supression
	# Fininding the Local Maxima
	r_min=np.amin(r)
	local_max=np.array([])
	local_max_x=np.array([])
	local_max_y=np.array([])

	for i in range(11,image.shape[0]-11): # disregarding the boundary 11 pixels
		for j in range(11,image.shape[1]-11): # disregarding the boundary 11 pixels
			temp = r[i,j]
			r[i,j]=r_min # replacing the value with r_min to prevent checking with itself
			if (temp>np.amax(r[i-1:i+2,j-1:j+2])) and (temp>8e-08): # threshold to prevent flat regions from being detected
				local_max=np.append(local_max,temp)
				local_max_x=np.append(local_max_x,j)
				local_max_y=np.append(local_max_y,i)
			r[i,j]=temp

	ind =np.argsort(-local_max) # sort in decending order
	r_localmax=local_max[ind]
	x_feat=local_max_x[ind]
	y_feat=local_max_y[ind]
	x=np.array([x_feat[0]]) # contains global max coordinates as the are not supressed by any
	y=np.array([y_feat[0]])
	n=1                     # since it already as global max

	for i in range(1,r_localmax.shape[0]):
		tempx=(x_feat-x_feat[i])**2
		tempy=(y_feat-y_feat[i])**2
		temp_radius2=tempx+tempy
		if ((np.amin(temp_radius2[0:i]))**0.5>3): # Checking for the minimum supression radius
			x=np.append(x,x_feat[i])
			y=np.append(y,y_feat[i])
			n=n+1
			if (n==5000):
				break

	confidences, scales, orientations = None, None, None
	return x,y, confidences, scales, orientations