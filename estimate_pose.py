import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def P_2D_3D(u,v,d,K):

    u0 = K[0][2]
    v0 = K[1][2]
    fy = K[1][1]
    fx = K[0][0]

    x = (u-u0)*d/fx
    y = (v-v0)*d/fy
    return (x,y,d)


def feature_match(img1, img2):
   r''' Find features on both images and match them pairwise
   '''
   max_n_features = 1000
   # max_n_features = 500
   use_flann = False # better not use flann

   detector = cv2.xfeatures2d.SIFT_create(max_n_features)

   # find the keypoints and descriptors with SIFT
   kp1, des1 = detector.detectAndCompute(img1, None)
   kp2, des2 = detector.detectAndCompute(img2, None)
   if (des1 is None) or (des2 is None):
      return [], []
   des1 = des1.astype(np.float32)
   des2 = des2.astype(np.float32)

   if use_flann:
      # FLANN parameters
      FLANN_INDEX_KDTREE = 0
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)
      matches = flann.knnMatch(des1,des2,k=2)
   else:
      matcher = cv2.DescriptorMatcher().create('BruteForce')
      matches = matcher.knnMatch(des1,des2,k=2)
   good = []
   pts1 = []
   pts2 = []
   try:
       # ratio test as per Lowe's paper
       for i,(m,n) in enumerate(matches):
          if m.distance < 0.75*n.distance:
             good.append([m])
             pts2.append(kp2[m.trainIdx].pt)
             pts1.append(kp1[m.queryIdx].pt)

       pts1 = np.int32(pts1)
       pts2 = np.int32(pts2)
       #print("Num matches = ", len(good))

       # cv.drawMatchesKnn expects list of lists as matches.
       #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
       #plt.imshow(img3),plt.show()

   except ValueError:
       print(matches)
       return pts1, pts2

   return pts1, pts2


def get_pose(rgb,depth,rgb_near,K):
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_near_gray = cv2.cvtColor(rgb_near, cv2.COLOR_RGB2GRAY)

    pts2d_curr, pts2d_near = feature_match(rgb_gray,rgb_near_gray)


    #dilating depth
    kernel = np.ones((4,4), np.uint8)
    depth_dilated = cv2.dilate(depth,kernel)

    # extract 3d pts
    pts3d_curr = []
    pts2d_near_filtered = [] # keep only feature points with depth in the current frame
    for i, pt2d in enumerate(pts2d_curr):
       # print(pt2d)
       u, v = pt2d[0], pt2d[1]
       z = depth_dilated[v, u]
       if z > 0:
          xyz_curr = P_2D_3D(u, v, z, K)
          pts3d_curr.append(xyz_curr)
          pts2d_near_filtered.append(pts2d_near[i])

    # the minimal number of points accepted by solvePnP is 4:
    if len(pts3d_curr)>=4 and len(pts2d_near_filtered)>=4:
       pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32), axis=1)
       pts2d_near_filtered = np.expand_dims(np.array(pts2d_near_filtered).astype(np.float32), axis=1)

       # ransac
       ret = cv2.solvePnPRansac(pts3d_curr, pts2d_near_filtered, K, distCoeffs=None)
       success = ret[0]
       rotation_vector = ret[1]
       translation_vector = ret[2]
       return (success, rotation_vector, translation_vector)
    else:
       return (0, None, None)


