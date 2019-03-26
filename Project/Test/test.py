import cv2

img = cv2.imread('/Users/brian/MyProjects/ECG_Project/yyyyyyy/7-1.jpg')
cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
