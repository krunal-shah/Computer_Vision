import numpy as np
import cv2


def get_points(img):
    points = [];
    img_to_show = img.copy()
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print "there"
            cv2.circle(img_to_show,(x,y),2,(255,0,0),-1)
            points.append([x,y])
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img_to_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return points

img = cv2.imread('img2.jpg')
# img = cv2.resize(img, (384,512))
points = get_points(img)
print points
pt1=np.asarray(points,dtype=np.float32)
print pt1.shape
size = points[1][0] - points[0][0]
new_points = np.array([[points[0][0],points[0][1]],
    [points[0][0]+size,points[0][1]],
    [points[0][0],points[0][1]+size],
    [points[0][0]+size,points[0][1]+size]])
pt2 = np.asarray(new_points, dtype=np.float32)
M = cv2.getPerspectiveTransform(pt1, pt2)
dst=cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
# cv2.imshow('image', dst)
# cv2.waitKey(0)
cv2.destroyAllWindows()

x = np.concatenate((pt1,np.ones([4,1])), axis=1)

# Two parallel lines and their point of intersection
l1 = np.cross(x[0],x[2])
l2 = np.cross(x[1],x[3])
pt1 = np.cross(l1, l2)

# Two parallel lines and their point of intersection
l3 = np.cross(x[0],x[1])
l4 = np.cross(x[2],x[3])
pt2 = np.cross(l3, l4)

# Line at infinity passing through the two points at infinity
linf = np.cross(pt1, pt2)
H = np.array([[1, 0, 0],[0, 1, 0],[linf[0]/linf[2], linf[1]/linf[2], 1]]);

dst1=cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))

con_img = np.concatenate((img, dst, dst1), axis=1)
cv2.imwrite('result3.jpg', con_img)
cv2.imshow('image', con_img)
cv2.waitKey(0)
cv2.destroyAllWindows()











