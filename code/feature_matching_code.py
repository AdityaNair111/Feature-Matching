import numpy as np
import time

def match_features(features1, features2, x1, y1, x2, y2):
    matches1 = np.array([[],[]])
    confidences1 = np.array([])
    matches2 = np.array([[],[]])
    confidences2 = np.array([])
    matches = np.array([[],[]])
    confidences = np.array([])
    r_threshold=0.8
    # Matchinf features from image 1 to image 2
    for if1 in range(features1.shape[0]):
        tempf1=features1[if1,:]*np.ones(features2.shape)
        temp_dist=(features2-tempf1)**2
        dist=np.sum(temp_dist,axis=1)
        ind=np.argsort(dist)
        dist=dist[ind]

        if (dist[1]!=0) and (((dist[0]/dist[1])**0.5)<r_threshold): # checking dist[1] to prevent dividing by zero

            matches1=np.append(matches1,np.array([[if1],[ind[0]]]),axis=1)
            c = np.array((dist[0]/dist[1]))
            confidences1 =np.append(confidences1,c)

    #Matching features from image 2 to image 1
    for if2 in range(features2.shape[0]):
        tempf2=features2[if2,:]*np.ones(features1.shape)
        temp_dist=(features1-tempf2)**2
        dist=np.sum(temp_dist,axis=1)
        ind=np.argsort(dist)
        dist=dist[ind]

        if (dist[1]!=0) and (((dist[0]/dist[1])**0.5)<r_threshold):

            matches2=np.append(matches2,np.array([[ind[0]],[if2]]),axis=1)
            c = np.array((dist[0]/dist[1]))
            confidences2 =np.append(confidences2,c)

    # Looking for Symmetric Matches that occured in both cases
    for i in range(matches1.shape[1]):
        for j in range(matches2.shape[1]):
            if (matches1[0,i]==matches2[0,j]) and (matches1[1,i]==matches2[1,j]):
                matches=np.append(matches,np.array([[matches1[0,i]],[matches1[1,i]]]),axis=1)
                if confidences1[i]>confidences2[j]: #selecting the lower of the two confidences
                    confidences=np.append(confidences,confidences1[i])
                else :
                    confidences=np.append(confidences,confidences2[j])

    confident=np.argsort(confidences) #sorting by most confident matches (smaller confidence score means its a better match)
    confidences=confidences[confident]
    matches0=matches[0,:]
    matches1=matches[1,:]
    matches[0,:]=matches0[confident]
    matches[1,:]=matches1[confident]    
    matches=np.transpose(matches)
    matches=matches.astype(int)
    return matches, confidences