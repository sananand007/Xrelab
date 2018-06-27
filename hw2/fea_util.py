import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
from skimage.feature import hog
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# load data from PIE dataset

def get_data():
    data = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in ['05','07','09','27','29']:
        data.append(sio.loadmat('PIE/Pose'+i+'_64x64.mat'))

    for i in range(len(data)):
        for j in range(data[i]['fea'].shape[0]):
            if data[i]['isTest'][j][0] < 1.0:
                train_data.append(data[i]['fea'][j].reshape(64,64))
                train_label.append(int(data[i]['gnd'][j]))
            else:
                test_data.append(data[i]['fea'][j].reshape(64,64))
                test_label.append(int(data[i]['gnd'][j]))
    return train_data, train_label, test_data, test_label

# function for visualizing facial  hog feauress
def hog_show(img, orient, pix_per_cell, cell_per_block):
    fea, img_show = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=True,
                               feature_vector=False)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(img_show, 'gray')
    plt.title('HOG')
    plt.show()

# function for extracting hog features using skimage.feature

def get_hog_feature(x_train,x_test, orient, pix_per_cell, cell_per_block):
    train_hog_feature = []
    test_hog_feature = []
    for i in range(len(x_train)):
        img = x_train[i]
        hi=hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False,feature_vector=False)
 	#print(hi.shape)

        train_hog_feature.append(hi)
    for j in range(len(x_test)):
        img = x_test[j]
        test_hog_feature.append(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False,feature_vector=False))
    return train_hog_feature, test_hog_feature

# function for extracting histogram of intensity
##
from collections import Counter
def get_pixel_range(key, ranges):
    # ranges, main=[], 255
    # if nBins==16:
    #     #ranges=[(0,15),(16,31),(32,47),(48,63),(64,79),(80,95),(96,111),\
    #     #(112,127),(128,143),(144,159),(160,175),(176,191),(192,207),(208,223),(224,239),(240,255)]
    #     #jump=round(float(nBins/255),4)
    #     jump=nBins
    #     count=0
    #     while main>0:
    #         ranges.append((count, count+jump))
    #         count+=jump
    #         main-=nBins
    # if nBins==9:
    #     ranges=[(0,28),(29,57),(58,86),(87,115),(116,144),(145,173),(174,202),(203,231),(232,255)]
    # if nBins>=10 and nBins<90:
    #     jump=20
    #     for x in range(0,256,jump):
    #         ranges.append((x,x+jump-1))
    # if nBins>=90:
    #     jump=2
    #     for x in range(0,256,jump):
    #         ranges.append((x,x+jump-1))
    for indx,range_ in enumerate(ranges):
        if key>=range_[0] and key<=range_[1]:
            return indx
    return -1

def get_intensity_feature(x_train,x_test, nBins):
    train_intensity_feature = []
    test_intensity_feature = []
    list_of_ranges_train=[0]*nBins
    list_of_ranges_test=[0]*nBins
    #print(x_train)
    jump=nBins
    ranges, main=[], 255
    count=0
    while main>0:
        ranges.append((count, count+jump))
        count+=jump
        main-=nBins
    for i in range(len(x_train)):
        img = x_train[i]
        flat_img=[item for sublist in img for item in sublist]
        c=Counter(flat_img)
        for key, val in c.items():
            #pixel=round(float(key/255),4)
            list_of_ranges_train[get_pixel_range(key, ranges)]+=val
        #hi=np.random.rand(nBins,1)
        #hi=[round(np.random.uniform(0,1),3) for i in range(nBins)]
        train_intensity_feature.append(list_of_ranges_train)
        list_of_ranges_train=[0]*nBins
        #train_intensity_feature.append(hi)
    for j in range(len(x_test)):
        img = x_test[j]
        flat_img=[item for sublist in img for item in sublist]
        c=Counter(flat_img)
        for key, val in c.items():
            #pixel=round(float(key/255),4)
            list_of_ranges_test[get_pixel_range(key, ranges)]+=val
        #hj=np.random.rand(nBins,1)
        #hj=[round(np.random.uniform(0,1),3) for i in range(nBins)] #random case
        test_intensity_feature.append(list_of_ranges_test)
        list_of_ranges_test=[0]*nBins
        #test_intensity_feature.append(hj)
    return train_intensity_feature, test_intensity_feature

############## function for extracting histogram of LM Filter responses
# x_train: training images
# x_test: testing images
# F: [49,49, 48] filters
# dimention of histogram is 48

def get_filter_feature(x_train,x_test, F):
    train_filter_feature = []
    test_filter_feature = []
    for i in range(len(x_train)):
        img = x_train[i]
	###############PLACEHOLDER START########################
	#Extracting the average absolute responses of each filter in the LM Bank
	# hi=np.random.rand(48,1)
	# ###############PLACEHOLDER END###########################
    #
    # train_filter_feature.append(hi)
    # for j in range(len(x_test)):
    #     img = x_test[j]
	# ###############PLACEHOLDER START########################
	# #Extracting the average absolute responses of each filter in the LM Bank
	# hj=np.random.rand(48,1)
    #     ###############PLACEHOLDER END###########################
    # test_filter_feature.append(hj)

    return train_filter_feature, test_filter_feature
