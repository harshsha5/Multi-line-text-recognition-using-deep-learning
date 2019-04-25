import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt

# import ipdb

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    min_region_size = 100
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    image = skimage.color.rgb2gray(image)
    image = image / np.max(image)
    grayscale = image
    # ipdb.set_trace()
    image = skimage.filters.gaussian(image)
    #How are we denoising it?

    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(10))

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)

    # label image regions
    label_image = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(label_image, image=image)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(grayscale)
    # plt.show()

    regions = skimage.measure.regionprops(label_image)
    for region in regions:
    	if (region.area >= min_region_size):
    		bboxes.append(region.bbox)
    bboxes = np.asarray(bboxes)

    return bboxes, grayscale