# Following code works in Python 2.7
import sys
import cv2
import numpy as np
import glob

# For Feature extraction and clustering the Data
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

# Dump the model generated for inference 
import pickle

# Get Train and Test folder via Sys arguments
input_train_folder = sys.argv[1]
input_test_folder = sys.argv[2]

# Get train and test image list using glob function
train_image_list = glob.glob(input_train_folder+"/*.png")
test_image_list = sorted(glob.glob(input_test_folder+"/*.png"))

# Configure image params
image_width = 256
image_height = 256
channels = 3
batch_size = 32


# Get total features using IPCA
ipca= IncrementalPCA()
x_train = []
# Read images in batch
for index_i in range(0, len(train_image_list),batch_size):
    for index_j in range(index_i,index_i+batch_size):
        if (index_j >= len(train_image_list)):
            break
        train_image_name = train_image_list[index_j]
        img = cv2.imread(train_image_name)
        img_resize = cv2.resize(img,(image_width,image_height))
        img_1d = np.reshape(img_resize, [1,image_height*image_width*channels])
        # Fit the images
        if (index_j + 1)%batch_size == 0:
            x_train = np.concatenate((x_train, img_1d), axis=0)
            ipca.partial_fit(x_train)
            x_train = []
        else:
            if (index_j%batch_size == 0):
                x_train = img_1d
            else: 
                x_train = np.concatenate((x_train, img_1d), axis=0)


print ("First Level PCA completed")

# Get Features count based variance value
k=0
total= sum(ipca.explained_variance_)
current_sum=0
while current_sum/total < 0.98:
    current_sum += ipca.explained_variance_[k]
    k+=1
print ("Featuers : ",k)


# Fit the train data using total number of features
ipca= IncrementalPCA(n_components=k)
x_train = []
# Read images in batch
for index_i in range(0, len(train_image_list),batch_size):
    for index_j in range(index_i,index_i+batch_size):
        if (index_j >= len(train_image_list)):
            break
        train_image_name = train_image_list[index_j]
        img = cv2.imread(train_image_name)
        img_resize = cv2.resize(img,(image_width,image_height))
        img_1d = np.reshape(img_resize, [1,image_height*image_width*channels])
        # Fit the images
        if (index_j + 1)%batch_size == 0:
            x_train = np.concatenate((x_train, img_1d), axis=0)
            ipca = ipca.partial_fit(x_train)
            x_train = []
        else:
            if (index_j%batch_size == 0):
                x_train = img_1d
            else:
                x_train = np.concatenate((x_train, img_1d), axis=0)

print ("Second level PCA completed")

# Cluster the transformed data using kmeans clustering mini-batch
kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=batch_size)
x_train = []
# Read images in batch
for index_i in range(0, len(train_image_list),batch_size):
    for index_j in range(index_i,index_i+batch_size):
        if (index_j >= len(train_image_list)):
            break
        train_image_name = train_image_list[index_j]
        img = cv2.imread(train_image_name)
        img_resize = cv2.resize(img,(image_width,image_height))
        img_1d = np.reshape(img_resize, [1,image_height*image_width*channels])
        if (index_j + 1)%batch_size == 0:
            x_train = np.concatenate((x_train, img_1d), axis=0)
            # Generate Features using IPCA
            pca_train_data = ipca.fit_transform(x_train)
            # Fit extracted features using Kmean clustering
            kmeans = kmeans.partial_fit(pca_train_data)
            x_train = []
        else:
            if (index_j%batch_size == 0):
                x_train = img_1d
            else:
                x_train = np.concatenate((x_train, img_1d), axis=0)

print ("Kmeans clustering completed")


# Transform the test data using IPCA and test the extracted features using kmeans clustering
for index,test_image_name in enumerate(test_image_list):
    img = cv2.imread(test_image_name)
    img_resize = cv2.resize(img,(image_width,image_height))
    img_1d = np.reshape(img_resize, [1,image_height*image_width*channels])
    pca_test_data = ipca.transform(img_1d)
    result = kmeans.predict(pca_test_data)
    print (test_image_name+" = "+str(result[0]))

print ("Test set processing completed")

# Dump Kmeans model and IPCA model for inference
filename = "kmeansmodel.sav"
pickle.dump(kmeans, open(filename, 'wb'))
filename = "IPCA.sav"
pickle.dump(ipca, open(filename, 'wb'))


