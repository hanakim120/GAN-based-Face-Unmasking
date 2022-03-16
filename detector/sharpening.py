import cv2 

img1 = cv2.imread("./data/001.jpg") 

#kernel
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#dilation
dilation = cv2.dilate(img1, kernel3, iterations=1)

#sharpen
after_sharpen = cv2.fastNlMeansDenoisingColored(dilation,None,50,50,7,21)

#Save
cv2.imwrite('./results/001_iter1.jpg', dilation)


