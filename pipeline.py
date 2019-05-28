# ===================================================================================
# IMPORT USEFUL PACKAGES.
# ===================================================================================
# Importing useful packages.
import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
import cv2
import os


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    ### FUNCTION: Direction of the Gradient  - dir_threshold()
    ### inputs:
    ### outputs:
    ### Resources:
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    ### FUNCTION: Magnitude of the Gradient  - mag_thresh()
    ### inputs:
    ### outputs:
    ### Resources:
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

#画框
def find_lane_pixels(binary_warped):

    cv2.imshow("binary_warped",binary_warped)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # cv2.imdecode("out_img",out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])  #左分布
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  #右分布

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 8
    # Set the width of the windows +/- margin
    margin = 20
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)  #计算 框 高度
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height  #上一个框
        win_y_high = binary_warped.shape[0] - window*window_height  #下一个框
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &  (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        # 计算是否需要偏移，偏移多少
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    return out_img



def fit_scanning(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # cv2.imshow("binary_warped", binary_warped)
    srcImg = np.copy(binary_warped)
    srcImg.fill(.0)

    # srcImg = np.zeros(binary_warped.shape, np.float64)
    # srcImg.fill(0.)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    good_inds = 500  #检测点

    nwindows = 8  #框高度 个数

    im_mean = binary_warped.shape[1] // 2  #图片中间点


    window_width = 30    #
    window_height = np.int(binary_warped.shape[0]//nwindows)  #计算 框 高度

    for window in range(nwindows):
        #  X  初始
        win_xleft_high = im_mean
        win_xright_low = im_mean

        #  Y
        win_y_low = out_img.shape[0] - (window + 1) * window_height  # 上一个框
        win_y_high = out_img.shape[0] - window * window_height  # 下一个框

        cv2.line(out_img, (im_mean, win_y_low), (im_mean, win_y_high), thickness=5, color=(0, 0, 255))

        left = 0
        right = 0
        while  win_xleft_high + window_width < out_img.shape[1]:
            win_xleft_low = win_xleft_high
            win_xleft_high += window_width

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
            left += 1
            if len(good_left_inds) > good_inds:
                break;


        while win_xright_low - window_width > 0:
            win_xright_high = win_xright_low
            win_xright_low = win_xright_high - window_width

            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            right += 1
            if len(good_right_inds) > good_inds:
                break;

        #中心偏移
        if right > left and np.int(right/left) < 3:
            im_mean -= (1 + np.int(right - ((right + left) / 2)))* window_width
        elif right < left and np.int(left/right) < 3:
            im_mean += (1 + np.int(left - ((right + left) / 2))) * window_width



        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xleft_high, win_y_high), (250, 0, 0), 2)

        #ROI
        roi = binary_warped[win_y_low:win_y_high, win_xright_low:win_xleft_high]
        srcImg[win_y_low:win_y_high, win_xright_low:win_xleft_high] = roi

    cv2.imshow("out_binary_warped",out_img)
    # cv2.imshow("srcImg", srcImg)
    return srcImg



#俯视图
def fit_cc(img,H_x,A_W,A_H,C_W):
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

#特征检测
def fit_binary(frame_undistort):
    # frame_undistort = cv2.undistort(frame, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(frame_undistort, orient='x', thresh_min=50, thresh_max=100)
    grady = abs_sobel_thresh(frame_undistort, orient='x', thresh_min=50, thresh_max=100)
    mag_binary = mag_thresh(frame_undistort, sobel_kernel=3, mag_thresh=(40, 100))
    dir_binary = dir_threshold(frame_undistort, sobel_kernel=3, thresh=(0.7, 1.3))
    combined_frame = np.zeros_like(dir_binary)
    combined_frame[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    image = cv2.cvtColor(frame_undistort, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined_frame)
    combined_binary[(s_binary == 1) | (combined_frame == 1)] = 1
    cv2.imshow('fit_binary', combined_binary)
    return combined_binary

def process_video (frame):

    #特征检测
    combined_binary = fit_binary(frame)

    #俯视图
    # combined_binaryxx = fit_cc(combined_binary, 0.61, 218, 716, 1227) #project_video
    combined_binaryxx = fit_cc(combined_binary, 0.64, 425, 1152, 2590)   #harder_challenge_video
    # combined_binaryxx = fit_cc(combined_binary, 0.68, 224, 925, 1029)   #challenge_video

    #扫描感情去部分
    binary_warped = fit_scanning(combined_binaryxx)
    # cv2.imshow('binary_warped', binary_warped)
    #选择线
    out_img = find_lane_pixels(binary_warped)
    cv2.imshow('out_img', out_img)
    return out_img


cap = cv2.VideoCapture('/root/视频/harder_challenge_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out_img = process_video (frame)
        cv2.imshow('frame', frame)
        cv2.imshow('out_img', out_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()