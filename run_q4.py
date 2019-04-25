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

def create_dataset(bboxes,bw):
    X = None
    for i in range (bboxes.shape[0]):    
        minr, minc, maxr, maxc = bboxes[i,:]
        array = bw[minr:maxr,minc:maxc]             #See if this is supposed to be minc:maxc...
        array = skimage.transform.resize(array, (32, 32),anti_aliasing=True)                            #See if anti aliasing is required

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.imshow(array)
        # plt.show()

        flattened_array = (np.transpose(array)).flatten()                   #Check!
        if(i==0):
            X = flattened_array
        else:
            X = np.vstack((X,flattened_array))

    return X

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

    X = create_dataset(bboxes,bw)

    h1 = forward(X,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    predicted_outputs = np.argmax(probs, axis=1)
    print(predicted_outputs)
    
