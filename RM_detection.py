# #opencv模板匹配----单目标匹配
# import cv2
#
# target = cv2.imread("data/target.jpg")
# template = cv2.imread("data/template.jpg")
#
# theight, twidth = template.shape[:2]
# result = cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED)
# #归一化处理
# cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# #匹配值转换为字符串
# strmin_val = str(min_val)
# cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
# #显示结果,并将匹配值显示在标题栏上
# cv2.imshow("MatchResult----MatchingValue="+strmin_val,target)
# cv2.waitKey()
# cv2.destroyAllWindows()


# #opencv----特征匹配----BFMatching
# import cv2
# from matplotlib import pyplot as plt
# #读取需要特征匹配的两张照片，格式为灰度图。
# template=cv2.imread("data/template_adjst.jpg",0)
# target=cv2.imread("data/target.jpg",0)
# orb=cv2.ORB_create()#建立orb特征检测器
# kp1,des1=orb.detectAndCompute(template,None)#计算template中的特征点和描述符
# kp2,des2=orb.detectAndCompute(target,None) #计算target中的
# bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) #建立匹配关系
# mathces=bf.match(des1,des2) #匹配描述符
# mathces=sorted(mathces,key=lambda x:x.distance) #据距离来排序
# result= cv2.drawMatches(template,kp1,target,kp2,mathces[:40],None,flags=2) #画出匹配关系
# plt.imshow(result),plt.show() #matplotlib描绘出来


# # 基于FLANN的匹配器(FLANN based Matcher)
# import cv2
# from matplotlib import pyplot as plt
# queryImage=cv2.imread("data/template_adjst.jpg",0)
# trainingImage=cv2.imread("data/target.jpg",0)#读取要匹配的灰度照片
# # sift=cv.xfeatures2d.SIFT_create()#创建sift检测器
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(queryImage,None)
# kp2, des2 = sift.detectAndCompute(trainingImage,None)
# #设置Flannde参数
# FLANN_INDEX_KDTREE=0
# indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
# searchParams= dict(checks=50)
# flann=cv2.FlannBasedMatcher(indexParams,searchParams)
# matches=flann.knnMatch(des1,des2,k=2)
# #设置好初始匹配值
# matchesMask=[[0,0] for i in range (len(matches))]
# for i, (m,n) in enumerate(matches):
# 	if m.distance< 0.5*n.distance: #舍弃小于0.5的匹配结果
# 		matchesMask[i]=[1,0]
# drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0) #给特征点和匹配的线定义颜色
# resultimage=cv2.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams) #画出匹配的结果
# plt.imshow(resultimage,)
# plt.show()


# # 基于FLANN的匹配器(FLANN based Matcher)定位图片
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
# template = cv2.imread('data/template_adjst.jpg', 0)  # queryImage
# target = cv2.imread('data/target.jpg', 0)  # trainImage
# # Initiate SIFT detector创建sift检测器
# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(template, None)
# kp2, des2 = sift.detectAndCompute(target, None)
# # 创建设置FLANN匹配
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# # 舍弃大于0.7的匹配
# for m, n in matches:
# 	if m.distance < 0.7 * n.distance:
# 		good.append(m)
# if len(good) > MIN_MATCH_COUNT:
# 	# 获取关键点的坐标
# 	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # shape(33,1,2)
# 	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# 	# 计算变换矩阵和MASK
# 	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# 	matchesMask = mask.ravel().tolist()
# 	h, w = template.shape
# 	# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
# 	pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# 	dst = cv2.perspectiveTransform(pts, M)
# 	cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
# else:
# 	print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
# 	matchesMask = None
# draw_params = dict(matchColor=(0, 255, 0),
# 				   singlePointColor=None,
# 				   matchesMask=matchesMask,
# 				   flags=2)
# result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
# plt.imshow(result, 'gray')
# plt.show()

# 基于ORB算法编写匹算法
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('data/template_adjst.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/target.jpg', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
# sift=cv2.xfeatures2d.SIFT_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], None, flags=2)
matches = bf.knnMatch(des1, des2, k=1)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
plt.imshow(img3), plt.show()

