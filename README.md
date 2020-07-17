# Indoor-outdoor-classification
Classify INDOOR and OUTDOOR using Unsupervised Learning.

Used IPCA                             ==> To extract features from the images

Used Kmeans mini batch clustering     ==> To cluster the extracted features

Requirements:
Following python packages are required to run the setup,
1) numpy
2) opencv-python
3) scikit

Code tested in Python 2.7 version.

How to run the code:

python Train_PCA_kmeans.py <Train folder with png images> <Test folder with png images>
  
IPCA model and Kmeans Model will be dumped at the final stage of execution. 
