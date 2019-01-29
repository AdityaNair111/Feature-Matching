import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    assert image.ndim == 2, 'Image must be grayscale'
    # caluculating feature gradient
    gx = cv2.Sobel(image,cv2.CV_64F,1,0,1)
    gy = cv2.Sobel(image,cv2.CV_64F,0,1,1)
    g = (((gx)**2)+((gy)**2))**0.5
    theta = np.arctan2(gy, gx) * 180 / np.pi
    
    feat_dim = 128
    x1 = np.round_(x)
    y1 = np.round_(y)
    k = int(feature_width) # Size of feature
    neighbours=1           # pixels from specified neighbourhood can contribute to the histogram of orientations
    weights = cv2.getGaussianKernel(ksize=k+2*neighbours,sigma=(k)*0.4) # gaussian window weighting
    weights = np.dot(weights,weights.transpose())
    fv=np.zeros((int(x1.shape[0]),(int(feat_dim))))

    for index in range(x1.shape[0]):
        i=y1[index];
        j=x1[index]
        window_g = g[int(i-k/2-neighbours):int(i+k/2+neighbours),int(j-k/2-neighbours):int(j+k/2+neighbours)]*weights
        window_theta = theta[int(i-k/2-neighbours):int(i+k/2+neighbours),int(j-k/2-neighbours):int(j+k/2+neighbours)]*weights
        f=np.array([])
            
        for m in range(0,k,int(k/4)):
            for n in range(0,k,int(k/4)):
                g_flat = (window_g[m:m+int(k/4+2*neighbours),n:n+int(k/4+2*neighbours)]).flatten()
                theta_flat = (window_theta[m:m+int(k/4+2*neighbours),n:n+int(k/4+2*neighbours)]).flatten()
                hist = np.zeros(8)
                for z in range(theta_flat.shape[0]):
                    if theta_flat[z]<0:
                        theta_flat[z]=theta_flat[z]+360
                        
                    if theta_flat[z]<45:
                        hist[0]=hist[0]+np.cos((theta_flat[z]) *np.pi/180)*g_flat[z]
                        hist[1]=hist[1]+np.sin((theta_flat[z]+ 45) *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=45) and (theta_flat[z]<90):
                        hist[1]=hist[1]+np.cos((theta_flat[z]- 45) *np.pi/180 )*g_flat[z]
                        hist[2]=hist[2]+np.sin(theta_flat[z]*np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=90) and (theta_flat[z]<135):
                        hist[2]=hist[2]+np.cos((theta_flat[z]- 90) *np.pi/180)*g_flat[z]
                        hist[3]=hist[3]+np.sin((theta_flat[z]- 45) *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=135) and (theta_flat[z]<180):
                        hist[3]=hist[3]+np.cos((theta_flat[z]- 135) *np.pi/180)*g_flat[z]
                        hist[4]=hist[4]+np.sin((theta_flat[z]- 90)  *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=180) and (theta_flat[z]<225):
                        hist[4]=hist[4]+np.cos((theta_flat[z]- 180) *np.pi/180)*g_flat[z]
                        hist[5]=hist[5]+np.sin((theta_flat[z]- 135) *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=225) and (theta_flat[z]<270):
                        hist[5]=hist[5]+np.cos((theta_flat[z]- 225) *np.pi/180)*g_flat[z]
                        hist[6]=hist[6]+np.sin((theta_flat[z]- 180) *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=270) and (theta_flat[z]<315):
                        hist[6]=hist[6]+np.cos((theta_flat[z]- 270) *np.pi/180)*g_flat[z]
                        hist[7]=hist[7]+np.sin((theta_flat[z]- 225) *np.pi/180)*g_flat[z]

                    elif (theta_flat[z]>=315) and (theta_flat[z]<360):
                        hist[7]=hist[7]+np.cos((theta_flat[z]- 315) *np.pi/180)*g_flat[z]
                        hist[0]=hist[0]+np.sin((theta_flat[z]- 270) *np.pi/180)*g_flat[z]

                f=np.append(f,hist,0)
        if np.any(f):                       #so that the code is not dividing by zero
            f = (f/np.linalg.norm(f,ord=2)) # normalize
        fv[index,:]=f
    fv=fv**1                       # raised to a power less than one to accentuate small values of features
    return fv