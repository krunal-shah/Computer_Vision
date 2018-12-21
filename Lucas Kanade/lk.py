# Invsere Compositional Algorithm for image stabilization

import numpy as np
import cv2



def lk(target_img):
	target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
	target_img = target_img.astype('float32')
	
	# Initialization of the parameters
	d = np.array([1,0,0,1,0,0]).astype('float32')
	warp_template = np.ones(np.shape(template))
	update = np.ones([6, 1])
	print "Entering while loop",
	i=0

	while (np.linalg.norm(update, axis = 0) > 0.001) and i<2500:
		# Calculate the warp image using the old parameters
		warp_template = cv2.warpAffine(target_img, np.transpose(np.reshape(d,[3,2])), (template.shape[1],template.shape[0]))
		# Compute the error image
		error = warp_template - template
		# Compute expression 7
		expr = np.sum(np.sum(np.matmul(np.transpose(steepest_d_images, (0,1,3,2)), np.expand_dims(np.expand_dims(error, axis=-1),axis=-1)), axis=0), axis=0)
		# Compute the update parameters using equation 37
		update = np.matmul(hessian_inv, expr)
		# Update d
		dnew = np.ones([6,1])
		dtemp = d
		# print np.transpose(d)
		dnew[0] = dtemp[0] + update[0] + dtemp[0]*update[0] + dtemp[2]*update[1]
		dnew[1] = dtemp[1] + update[1] + dtemp[1]*update[0] + dtemp[3]*update[1]
		dnew[2] = dtemp[2] + update[2] + dtemp[0]*update[2] + dtemp[2]*update[3]
		dnew[3] = dtemp[3] + update[3] + dtemp[1]*update[2] + dtemp[3]*update[3]
		dnew[4] = dtemp[4] + update[4] + dtemp[0]*update[4] + dtemp[2]*update[5]
		dnew[5] = dtemp[5] + update[5] + dtemp[1]*update[4] + dtemp[3]*update[5]
		# print str(i) + " " + str(np.linalg.norm(update, axis = 0))
		d = dnew
		i = i + 1
		if i%100==0:
			print i,

	# Resultant image from LK
	# final_img = cv2.warpAffine(target_img, np.transpose(np.reshape(d,[3,2])), (template.shape[1],template.shape[0]))
	print 
	return np.transpose(np.reshape(d,[3,2]))


cap = cv2.VideoCapture('video1.MOV')

frames = 0
ret, initial_frame = cap.read()
initial_frame = cv2.resize(initial_frame, (480,270))

# r = cv2.selectROI("Select",img=initial_frame,fromCenter=False, showCrosshair=False)
# r = initial_frame[r[1]:r[1]*r[3],r[0]:r[0]*r[2]]
# cv2.destroyAllWindows()

# template = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

template = template.astype('float32')

dtdx = cv2.Sobel(template, cv2.CV_32F, 1, 0)
dtdy = cv2.Sobel(template, cv2.CV_32F, 0, 1)
t_h = np.shape(template)[0]
t_w = np.shape(template)[1]

# Computing the derivative matrix of the warp function with respect to the parameters
coord = np.indices([t_h, t_w])
temp = np.copy(coord[0,:,:])
coord[0,:,:] = coord[1,:,:]
coord[1,:,:] = temp
coord  = np.concatenate([coord,np.expand_dims(np.ones([t_h, t_w]), axis =0)], axis=0)
coord = np.moveaxis(coord, 0, -1)
temp_matr = np.array([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0]])
temp_matr1 = np.array([[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]])
temp_matr = np.broadcast_to(temp_matr, [t_h, t_w, 6, 3])
temp_matr1 = np.broadcast_to(temp_matr1, [t_h, t_w, 6, 3])
dwdp_0 = np.matmul(temp_matr,np.expand_dims(coord, axis=3))
dwdp_1 = np.matmul(temp_matr1,np.expand_dims(coord, axis=3))
dwdp = np.concatenate([dwdp_1,dwdp_0], axis = 3)
dwdp = np.transpose(dwdp, (0,1,3,2))

# Derivative matrix of the image
delta_T = np.stack([dtdy, dtdx], axis=2)
delta_T = np.expand_dims(delta_T, axis=2)
# Steepest descent images and Hessian computation
steepest_d_images = np.matmul(delta_T, dwdp)
hessian = np.matmul(np.transpose(steepest_d_images, (0,1,3,2)), steepest_d_images)
hessian = np.sum(hessian, axis = 0)
hessian = np.sum(hessian, axis=0)
hessian_inv = np.linalg.inv(hessian)

h = np.shape(initial_frame)[0]
w = np.shape(initial_frame)[1]
out = cv2.VideoWriter('output2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w*2 + 20,h))
frame = 0

while(cap.isOpened()):
	ret, new_frame = cap.read()
	print frame
	# if frame > 100:
	# 	break
	if ret==True:
		new_frame = cv2.resize(new_frame, (480,270))
		params = lk(new_frame)
		params1 = np.copy(params)
		# final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
		final_img = cv2.warpAffine(new_frame, params1, (initial_frame.shape[1],initial_frame.shape[0]))
		con_img = np.concatenate((new_frame,170*np.ones([new_frame.shape[0],20,3],dtype=np.uint8), final_img), axis=1)
		con_img = con_img.astype('uint8')
		cv2.imshow('image', con_img.astype('uint8'))
		# final_img = final_img.astype('uint8')
		out.write(con_img)
		frame = frame + 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()