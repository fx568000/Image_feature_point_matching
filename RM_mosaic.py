import numpy as np
import cv2 as cv
import imutils
from math import *

def stitch(img1,img2, ratio = 0.75, reprojThresh = 4.0, showMatches = True):
    print('A')
    # (img2, img1) = imgs
    #获取关键点和描述符
    (kp1, des1) = detectAndDescribe(img1)
    (kp2, des2) = detectAndDescribe(img2)
    print(len(kp1),len(des1))
    print(len(kp2), len(des2))
    R = matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

    #如果没有足够的最佳匹配点，M为None
    if R is None:
        return  None
    (good, M, mask) = R
    print(M)
    #对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
    result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0])) # 重点2：先将img1按照透视矩阵进行曲面变形，再将曲面变形结果赋给result图
    # 重点3：再将图片2贴合在结果图片左端
    # result[0:img2.shape[0], 0:img2.shape[1]] = img2

    #是否需要显示ROI区域
    if showMatches:
        vis = drawMatches1(img1, img2, kp1, kp2, good, mask)
        return (result, vis)

    return result


def detectAndDescribe(img):
    print('B')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



    sift = cv.xfeatures2d.SIFT_create()
    (kps, des) = sift.detectAndCompute(img, None)

    kps = np.float32([kp.pt for kp in kps]) #    **********************************
    #返回关键点和描述符
    return (kps, des)

def matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh): # 描述符匹配器创建
    print('C')
    #初始化BF,因为使用的是SIFT ，所以使用默认参数
    matcher = cv.DescriptorMatcher_create('BruteForce') #
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    matches = matcher.knnMatch(des1, des2, 2)  #***********************************

    #获取理想匹配
    good = []
    for m in matches:
        if len(m) == 2 and  m[0].distance < ratio * m[1].distance:
            good.append((m[0].trainIdx, m[0].queryIdx))

    print(len(good))
    #最少要有四个点才能做透视变换，是因为4维数据决定的吗
    if len(good) > 4:
        #获取关键点的坐标
        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.float32([kp1[i] for (_, i) in good])
        dst_pts = np.float32([kp2[i] for (i, _) in good])

        #重点1：通过两个图像的关键点计算变换矩阵（透视矩阵）
        (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)

        #返回最佳匹配点、变换矩阵和掩模
        return (good, M, mask)
    #如果不满足最少四个 就返回None
    return None

def drawMatches(img1, img2, kp1, kp2, matches, mask, M):
    # 获得原图像的高和宽
    h, w = img1.shape[:2]
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得目标图像上对应的坐标
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor = (0, 255, 0),
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)
    img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    return img

def drawMatches1(img1, img2, kp1, kp2, metches,mask):
    print('D')
    (hA,wA) = img1.shape[:2]
    (hB,wB) = img2.shape[:2]
    vis = np.zeros((max(hA,hB), wA+wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2
    for ((trainIdx, queryIdx),s) in zip(metches, mask):
        if s == 1:
            ptA = (int(kp1[queryIdx][0]), int(kp1[queryIdx][1]))
            ptB = (int(kp2[trainIdx][0])+wA, int(kp2[trainIdx][1]))
            cv.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis

def image_pre(img1,img2):
    # opencv旋转图像且保持图像不被裁减
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    degree = 270
    heightNew1 = int(width1 * fabs(sin(radians(degree))) + height1 * fabs(cos(radians(degree))))
    widthNew1 = int(height1 * fabs(sin(radians(degree))) + width1 * fabs(cos(radians(degree))))
    heightNew2 = int(width2 * fabs(sin(radians(degree))) + height2 * fabs(cos(radians(degree))))
    widthNew2 = int(height2 * fabs(sin(radians(degree))) + width2 * fabs(cos(radians(degree))))
    matRotation1 = cv.getRotationMatrix2D((width1 / 2, height1 / 2), degree, 1)
    matRotation2 = cv.getRotationMatrix2D((width2 / 2, height2 / 2), degree, 1)
    matRotation1[0, 2] += (widthNew1 - width1) / 2
    matRotation1[1, 2] += (heightNew1 - height1) / 2
    matRotation2[0, 2] += (widthNew2 - width2) / 2
    matRotation2[1, 2] += (heightNew2 - height2) / 2
    img1 = cv.warpAffine(img1, matRotation1, (widthNew1, heightNew1), borderValue=(255, 255, 255))
    img2 = cv.warpAffine(img2, matRotation2, (widthNew2, heightNew2), borderValue=(255, 255, 255))
    # 确定固定尺寸
    img1 = imutils.resize(img1, width=400)
    img2 = imutils.resize(img2, width=400)
    return img1,img2

def show():
    img1 = cv.imread('data/img/50.png')
    img2 = cv.imread('data/img/850.png')

    img1,img2 = image_pre(img1,img2)

    # stitcher = cv.Stitcher()
    (result, vis) = stitch(img1,img2, showMatches=True)
    # (result, vis) = stitch(img1, img2)
    # result = stitch(img1, img2)

    cv.imshow('image A', img1)
    cv.imshow('image B', img2)
    cv.imshow('keyPoint Matches', vis)
    cv.imshow('Result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()
show()
