# import os
# import cv2
# from PIL import Image
#
#
# def unlock_mv(sp):
#     """ 将视频转换成图片
#         sp: 视频路径 """
#     cap = cv2.VideoCapture(sp)
#     suc = cap.isOpened()  # 是否成功打开
#     frame_count = 0
#     while suc:
#         frame_count += 1000 # 跳帧数
#         suc, frame = cap.read()
#         params = []
#         params.append(2)  # params.append(1)
#         cv2.imwrite(r'C:\Users\fengx\Desktop\福州黑臭水体\data\%d.jpg' % frame_count, frame, params)
#
#     cap.release()
#     print('unlock image: ', frame_count)
#
#
# def jpg2video(sp, fps):
#     """ 将图片合成视频. sp: 视频路径，fps: 帧率 """
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     images = os.listdir('mv')
#     im = Image.open('mv/' + images[0])
#     vw = cv2.VideoWriter(sp, fourcc, fps, im.size)
#
#     os.chdir('mv')
#     for image in range(len(images)):
#         # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
#         jpgfile = str(image + 1) + '.jpg'
#         try:
#             frame = cv2.imread(jpgfile)
#             vw.write(frame)
#         except Exception as exc:
#             print(jpgfile, exc)
#     vw.release()
#     print(sp, 'Synthetic success!')
#
#
# if __name__ == '__main__':
#     sp = "data/VID_20200417_111555.mp4"
#     sp_new = '识别.avi'
#     unlock_mv(sp)  # 视频转图片
#     # jpg2video(sp_new, 28)  # 图片转视频


import cv2
vc = cv2.VideoCapture('data/VID_20200417_111555.mp4') #读入视频文件
io = 'data/img/'
c=0
rval=vc.isOpened()
timeF = 50  #视频帧计数间隔频率
while rval:   #循环读取视频帧
    c = c + 1
    rval, frame = vc.read()
    if(c%timeF == 0): #每隔timeF帧进行存储操作
        cv2.imwrite(io+str(c) + '.png', frame) #存储为图像
        # cv2.imwrite('data/img/1.png', frame)
#     if rval:
# 	    #img为当前目录下新建的文件夹
#         cv2.imwrite('img/'+str(c) + '.jpg', frame) #存储为图像
#         cv2.waitKey(1)
#     else:
#         break
vc.release()

