import cv2
import os
from pathlib import Path
import numpy as np
import copy
import matplotlib as mtb
from matplotlib import pyplot as plt

folder_cnt=0
for l in range(2):
	for w in range(2):
		for n in range(12):
			for col in range(2):
				folder_cnt+=1
				class_name = str(l)+'_'+str(w)+'_'+str(n)+'_'+str(col)
				image_folder = os.path.dirname(os.path.realpath(__file__))
				image_folder+='/class '+ class_name
				print image_folder
				
				images = ([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
				
				cnt=1

				images_new=list()
				cur=list()
				prev = list()
				temp=list()
				print 'length = ', (len(images))
				# by this loop, first we concatenate 3 images horizontally 3 times and then concatenate those 3 images vertically
				# by this for every class we have resultant 10 images
				for image in images:
					if(cnt>90):
						print folder_cnt,' --- ', cnt
						break
					pathh=image_folder+'/'+image
					temp = cv2.imread(pathh)
					if len(cur)==0:
						cur=copy.copy(temp)
					else:		
						cur=np.concatenate((cur,temp),axis=1)
					# plt.imshow(cur)
					# plt.show()
					
					if(cnt%3==0):
						if len(prev)==0:
							prev = copy.copy(cur)
						else:
							prev=np.concatenate((prev,cur),axis=0)
						cur=[]

					if(cnt%9==0):
						x=cnt/9
						dir=os.path.dirname(os.path.realpath(__file__))+'/temp/'+class_name+'-'+str(x)+'.jpg'
						plt.imsave(dir,prev)
						prev=[]
						cur=[]
					cnt+=1;



# this is used to make video for all the images
video_name = 'video.avi'
print folder_cnt
video = cv2.VideoWriter(video_name, cv2.cv.CV_FOURCC('M','P','E','G'), 2, (28*3,28*3))
image_folder = os.path.dirname(os.path.realpath(__file__))+'/temp'
images_new = ([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

for image in images_new:
	video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()