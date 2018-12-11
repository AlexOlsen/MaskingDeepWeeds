# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:50:05 2018

@author: jc210083
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

image_directory = 'datasets\\deepweeds\\val'
annotation_file = 'datasets\\deepweeds\\deepweeds-lantana-val.json'

example_coco = COCO(os.path.join(ROOT_DIR, annotation_file))

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['lantana'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]

# load and display instance annotations
image = io.imread(os.path.join(ROOT_DIR, image_directory, image_data['file_name']))
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=False)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)