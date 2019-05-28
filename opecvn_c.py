import math

import cv2
import numpy as np

# dst_s = cc_(dst, 0.59, 217, 1347, 3592)  #直线

# img = cv2.imread('2019.png')

cap = cv2.VideoCapture('/root/视频/challenge_video.mp4')
ret, img = cap.read()

#图片  #要处理原图高度 0.0 ~ 1.0   #底宽   #高度   #顶宽
def cc_(img,H_x,A_W,A_H,C_W):
    H, W = img.shape[:2]

    H_x = math.ceil(H * H_x)
    pts1 = np.float32([[0, H], [W, H], [0, H_x], [W, H_x]])

    A_ = [(C_W / 2) - (A_W / 2), A_H]
    B_ = [(C_W / 2) + (A_W / 2), A_H]
    C_ = [0, 0]
    D_ = [C_W, 0]

    # 变换后分别在[A',B',C',D']
    pts2 = np.float32([A_, B_, C_, D_])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(img, M, (C_W, A_H))
    return dst

def nothing(x):
    # 原图中书本的四个角点[A,B,C,D]
    H_x = cv2.getTrackbarPos('h_x', 'frame')/100 #0.7  # 处理原图高度
    A_W = cv2.getTrackbarPos('A_W', 'frame')  #200
    A_H = cv2.getTrackbarPos('A_H', 'frame')  #1000
    C_W = cv2.getTrackbarPos('C_W', 'frame')  #2000
    dst = cc_(img,H_x,A_W,A_H,C_W)

    cv2.imshow('dst', dst)


cv2.namedWindow('frame')
cv2.imshow('frame',img)
cv2.createTrackbar('h_x','frame',40,100,nothing)  #要处理原图高度 0.0 ~ 1.0
cv2.createTrackbar('A_W','frame',200,1000,nothing)  #底宽
cv2.createTrackbar('A_H','frame',1000,5000,nothing)  #高度
cv2.createTrackbar('C_W','frame',2000,10000,nothing)  #顶宽

# c = cv2.getTrackbarPos('c', 'frame')  #获取值


cv2.waitKey(0)