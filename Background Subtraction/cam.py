import numpy as np
import cv2
import calck
from time import sleep
from scipy import misc
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=40)


k = 2
ret, frame = cap.read()
sleep(0.1)
ret, frame = cap.read()
[height, width, n] = np.shape(frame)
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
means = np.ones([k, height, width])
means[:,:,:] = gray_image  # k*height*width
omega = np.ones([k, height, width])
omega[0] = omega[0]*0.6
omega[1]= omega[1]*0.4
variances = np.ones([k, height, width])
variances[0] = variances[0]*1
variances[1] = variances[1]*30
rho = np.ones([k, height, width])
i = 3
sample_variance = np.ones([height, width])
sample_diff = np.ones([height, width])
print height
print width
while(True):
    # Capture frame-by-frame
    i = i+1
    alpha = i/3
    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    k_matrix = calck.calculatek(gray_image, means, variances)  #height*width
    new_img = np.copy(k_matrix)
    im = np.array(k_matrix * 255, dtype = np.uint8)
    #threshed = cv2.adaptiveThreshold(im, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    
    count_bin = np.bincount(np.reshape(new_img, [height*width]))
    k_matrix_flat = np.reshape(k_matrix, [height*width], order='F')
    indices_ = np.indices([height, width])
    height_indices = np.reshape(indices_[0], [height*width], order='F')
    width_indices = np.reshape(indices_[1], [height*width], order='F')
    count_bin = np.bincount(k_matrix_flat)
    
    #omega[k_matrix_flat, height_indices, width_indices] = (1-alpha)*omega[k_matrix_flat, height_indices, width_indices] + alpha*count_bin[k_matrix_flat]
    rho[k_matrix_flat, height_indices, width_indices] = (alpha)/omega[k_matrix_flat, height_indices, width_indices]#/(alpha)
    means[k_matrix_flat, height_indices, width_indices] = (1-rho[k_matrix_flat, height_indices, width_indices])*means[k_matrix_flat, height_indices, width_indices] + rho[k_matrix_flat, height_indices, width_indices]*gray_image[height_indices, width_indices]
    sample_diff[height_indices, width_indices] = gray_image[height_indices, width_indices] - means[k_matrix_flat, height_indices, width_indices] 
    sample_variance = np.multiply(sample_diff, sample_diff)
    variances[k_matrix_flat, height_indices, width_indices] = (1-rho[k_matrix_flat, height_indices, width_indices])*variances[k_matrix_flat, height_indices, width_indices] + rho[k_matrix_flat, height_indices, width_indices]*sample_variance[height_indices, width_indices]
    # a(k * height * width)
    # b(height * width)    (i,j)==>k
    # c(i,j) = a(k,i,j)


    fgmask = fgbg.apply(frame)
    #cv2.imshow('frame',fgmask)
    cv2.namedWindow( "frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow( "inbuilt", cv2.WINDOW_NORMAL)
    cv2.namedWindow( "result", cv2.WINDOW_NORMAL)
    cv2.imshow('frame1', gray_image)
    cv2.imshow('inbuilt', fgmask)
    cv2.imshow('result', im)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
    	break
	
	# # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Display the resulting frame
    # cv2.imshow('frame',gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()