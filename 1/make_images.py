import numpy as np
import matplotlib as mtb
from matplotlib import pyplot as plt
import cv2
import math
from PIL import Image, ImageDraw
import matplotlib.lines as mlines
import os
from shutil import copyfile


for i in range(1):
	n=3
	dir_path = os.path.dirname(os.path.realpath(__file__))
	print n
	length=15.0
	width=1
	angle=15*n
	color = (255,0,0)
	# color = (0,0,255)
	class_name = '1_0_'+str(n)+'_0'

	# this part of code is used to find starting and ending point of line
	ang=math.radians(angle)
	y=int(math.floor(length*math.cos(ang)))
	x=int(math.floor(length*math.sin(ang)))

	p1=(y,0)
	p2=(0,x)
	if(y<0):
		p1=(0,0)
		p2=(-y,x)
	if(x<0):
		p1=(y,-x)
		p2=(0,0)

	print p1,p2
	xx=p1
	xy=p2

	# this part of code is used to save files in particular folder
	dir_path += '/class '+ class_name 
	print dir_path 
	if not os.path.exists(dir_path):
	    os.makedirs(dir_path)
	cnt=1
	while(p1[1]<=27 and p2[1]<=27):
		while(p1[0]<=27 and p2[0]<=27):
			img = np.zeros((28,28,3), np.uint8)
			cv2.line(img,p2,p1,color,width)
			# print p1,p2
			p2=(p2[0]+1,p2[1])
			p1=(p1[0]+1,p1[1])
			# plt.imshow(img)
			# plt.show()
			plt.imsave(dir_path+'/'+class_name+'_'+str(cnt)+'.jpg',img)
			# break
			cnt+=1
		p1=(xx[0],p1[1]+1)
		p2=(xy[0],p2[1]+1)

	print cnt 


# this part of code is used to make replications of file to make 1000 images per file
# folders = os.listdir('./')
# print((folders))
# for folder in folders:
# 	if(folder.startswith('class')):
# 		# print folder
# 		files = (os.listdir('./'+folder+'/'))
# 		dir_path='./'+folder
# 		print folder, ' --- ', len(files)
# 		if(len(files)<1000):
# 			cnt = len(files)+1
# 			print cnt
# 			for i in range(len(files)):
# 				file = files[i]
# 				print file
# 				copyfile(dir_path+'/'+file,dir_path+'/'+class_name+'_'+str(cnt)+'.jpg')
# 				# break
# 				cnt+=1
# 				if(cnt>1000):
# 					break;
# 				if(i>=len(files)):
# 					i=0

# 				print cnt
