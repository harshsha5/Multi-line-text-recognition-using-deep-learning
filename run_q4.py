import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
# from matplotlib.patches import Circle
import ipdb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def create_dataset(bboxes,bw,input_size = 32):
    X = None
    for i in range (bboxes.shape[0]):    
        minr, minc, maxr, maxc = bboxes[i,:]
        array = bw[minr:maxr,minc:maxc]             #See if this is supposed to be minc:maxc...
        array = skimage.transform.resize(array, (input_size, input_size),anti_aliasing=True)                            #See if anti aliasing is required

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.imshow(array)
        # plt.show()

        flattened_array = (np.transpose(array)).flatten()                   #Check!

        if(i==0):
            X = flattened_array
        else:
            X = np.vstack((X,flattened_array))

    return X

def form_array_and_see_image(minr,maxr,minc,maxc,bw):
    array = bw[minr:maxr,minc:maxc]             #See if this is supposed to be minc:maxc...
    array = skimage.transform.resize(array, (32, 32),anti_aliasing=True)                            #See if anti aliasing is required

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(array)
    plt.show()

def find_indices_of_spaces(bboxes,average_word_length):
    factor_of_word_length = 2.1     #The factor by which the next character should be away from the previous to characterize it as a new word
    indices_of_spaces = []
    for i in range(bboxes.shape[0]-1):
        minr_prev, minc_prev, maxr_prev, maxc_prev = bboxes[i,:]  
        minr_next, minc_next, maxr_next, maxc_next = bboxes[i+1,:] 
        if(minc_next > minc_prev + factor_of_word_length*average_word_length):
            indices_of_spaces.append(i)
        #ipdb.set_trace()
    return indices_of_spaces

def order_bounding_boxes_as_rows(bboxes_original,bw):
    bboxes = bboxes_original[bboxes_original[:,1].argsort()] #Sort bounding boxes by column number. So that when we sort them by rows later, they are already sorted by column.
    count = 0
    dict_of_bboxes = {0 : [0]}
    cluster_properties = []
    sum_of_word_lengths = 0
    '''
    Logic: Find the coordinates of the centroid of a bounding box. If the next bounding box has a centroid within the minr and maxr(plus some margin)
    then we put it in the same class (ie. the samw row) else we add a new class. Once they are sorted according to rows, we order them to see which row
    comes first in row-wise by comparing the clusters properties (ie. minr of the various clusters)
    '''
    for i in range(bboxes.shape[0]):
        minr, minc, maxr, maxc = bboxes[i,:]
        x_centre = minc + (maxc-minc)/2
        y_centre = minr + (maxr-minr)/2
        sum_of_word_lengths += maxc-minc
        #form_array_and_see_image(minr,maxr,minc,maxc,bw)
        flag = 0
        if(i==0):
            cluster_properties.append((.95*minr,1.05*maxr))
        else:
            for j,elt in enumerate(cluster_properties):
                if(y_centre>elt[0] and y_centre<elt[1]):
                    dict_of_bboxes[j].append(i)
                    #print("Classified in class",j)
                    flag=1
                    break
            if(flag!=1):
               cluster_properties.append((.95*minr,1.05*maxr))
               index = len(cluster_properties)-1
               dict_of_bboxes[index] = [i] 
               #print("new row created")

    row_order_for_classes = sorted(range(len(cluster_properties)), key=lambda k: cluster_properties[k][0])  #Orders the indices according to minr. So lower the minr earlier your cluster is.
    
    reordered_bboxes = np.zeros((1,4))
    previous_length = 0
    indices_of_new_line = []
    for elt in row_order_for_classes:
        for element in dict_of_bboxes[elt]:
            reordered_bboxes = np.vstack((reordered_bboxes,bboxes[element,:]))
        previous_length += len(dict_of_bboxes[elt])
        indices_of_new_line.append(previous_length)     #If this array has j, that means jth word should be a new_line(consider zero indexed array) in the bounding box array

    reordered_bboxes = reordered_bboxes[1:,:]
    average_word_length = sum_of_word_lengths/reordered_bboxes.shape[0]
    print(reordered_bboxes)
    print(average_word_length)
    indices_of_spaces = find_indices_of_spaces(reordered_bboxes,average_word_length)


    '''
    Order them in height order, ie. Top row first then the next etc. Once this is done we can 
    save the indices of the spaces.
    If the distance between the start of next > start of previous + 1.3 mean_length_of_word_in_a_row then it's a different word
    '''
    ipdb.set_trace()

    return reordered_bboxes,indices_of_new_line,indices_of_spaces



#Referenced Tutorial for this question

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    # for bbox in bboxes:
    for i in range (bboxes.shape[0]):    
        minr, minc, maxr, maxc = bboxes[i,:]
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        # circle = Circle((maxc, maxr), 10, facecolor='none',edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        # plt.gca().add_patch(circle)
    plt.show()
    # print(bboxes.shape,type(bboxes))
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    bboxes = order_bounding_boxes_as_rows(bboxes,bw)
    # X = create_dataset(bboxes,bw)

    # h1 = forward(X,params,'layer1')
    # probs = forward(h1,params,'output',softmax)
    # predicted_outputs = np.argmax(probs, axis=1)
    # print(predicted_outputs)

    '''
    See how to output the final labels. See better characterisation of the bounding boxes. How to pick appropriate bounding box for sentence formation
    rather than randomly.
    '''
    
