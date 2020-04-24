# Face Detection With Haar Cascade alogrithm
   
   
This is a tutorial for using Haar Cascade, a machine learning object detection algorithm to locate and identify objects in images or videos.


# Libaries 
Python3
Opencv
numpy


* File Structure
 +datasetcreator.py
 +trainer.py
 +detect.py
 +face_recognition.py
 +face_from_image.py
 +face_from_video.py
 
 # Haar Cascade
 
Haar features are like kernels with specified weights, not like CNN's learned weights. is 

 Types  of Haar features below:

 ![alt text](/haar.jpg "Haar filters")

 

The first two behave like the Sobel features extracting horizontal and vertical edge, the slightly slanted kernel, can extract slanted line. 



***

Haar features applied to an image:



![alt text](/haar_applied.png "Haar filters")

***



The next stage is Adaboost:

Applying these multiple kernels over an image, its produces an enormous set of features. "Adaboost" is a method used to cut down this features. It picks out relevant features that at least help in classifying the object by finding a part of the object. These classifiers are known as weak classifiers, they are summed up using their weight factor "αᵢ" as seen below.
F( x ) = ∑ ( αᵢ * fᵢ( x ) )



***

The next stage is cascading:

![alt text](/Cascade.png "Cascading")



Using windows of different sizes, it scans through, checking for presence of faces. if not detected it is immediately dropped.



The learned weights is then saved. They are the important components of this files "haarcascade_frontalface_default.xml" and "haarcascade_eye.xml". Please click  for others. [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)


 
 "R.Leinhart, J.Maydt. An Extended Set of Haar-like Features for Rapid Object Detection. IEEE ICIP, vol.1, pp. 900-903, Sep. 2002."
 


