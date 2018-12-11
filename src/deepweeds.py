"""
Mask R-CNN
Train on the DeepWeeds dataset and implement lantana detector.
Written by Alex Olsen

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 deepweeds.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 deepweeds.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 deepweeds.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Test detection on a single image
    python3 deepweeds.py test --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import os.path
import sys
import numpy as np
import skimage.draw
import imgaug
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class DeepWeedsConfig(Config):
    """Configuration for training on the DeepWeeds dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "deepweeds"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Lantana

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 60 // IMAGES_PER_GPU
    
    VALIDATION_STEPS = 20 // IMAGES_PER_GPU
    
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512 #twice as many as default
    
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
    
    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 960
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 2.0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 50 # 1/4th of default


############################################################
#  Dataset
############################################################

class DeepWeedsDataset(utils.Dataset):

    def load_deepweeds(self, dataset_dir, subset, return_deepweeds=False):
        """Load a subset of the DeepWeeds dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        cocoDeepWeeds = COCO("{}\\deepweeds-lantana-{}.json".format(dataset_dir, subset))
        image_dir = os.path.join(dataset_dir, subset)
        
        # Visualise some info of dataset
        categories = cocoDeepWeeds.loadCats(cocoDeepWeeds.getCatIds())
        category_names = [category['name'] for category in categories]
        print('DeepWeeds categories: \n{}\n'.format(' '.join(category_names)))
        
        category_names = set([category['supercategory'] for category in categories])
        print('DeepWeeds supercategories: \n{}'.format(' '.join(category_names)))

        # Load all classes
        class_ids = sorted(cocoDeepWeeds.getCatIds())

        # Load all images
        image_ids = list(cocoDeepWeeds.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepweeds", i, cocoDeepWeeds.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "deepweeds", image_id=i,
                path=os.path.join(image_dir, cocoDeepWeeds.imgs[i]['file_name']),
                width=cocoDeepWeeds.imgs[i]["width"],
                height=cocoDeepWeeds.imgs[i]["height"],
                annotations=cocoDeepWeeds.loadAnns(cocoDeepWeeds.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=False)))
        
        if return_deepweeds:
            return cocoDeepWeeds
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_path = image_path.replace("bmp","png")

        # Read mask file from .png image
        mask = []
        m = skimage.io.imread(mask_path).astype(np.bool)
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "deepweeds":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

###############################################################################
# TRAINING
###############################################################################

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DeepWeedsDataset()
    dataset_train.load_deepweeds(args.dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DeepWeedsDataset()
    dataset_val.load_deepweeds(args.dataset_dir, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    #print("Training network heads")
    #model.train(dataset_train, dataset_val,
    #            learning_rate=config.LEARNING_RATE,
    #            epochs=10,
    #            layers='heads')
    
    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)

###################################################################
# TESTING
###################################################################


def visualise_mask(image, mask):
    """Colour the found mask region in the image
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    #gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    (height, width, dim) = image.shape
    red = np.zeros([height, width, dim], dtype=np.uint8)
    red[:, :, 0].fill(255)
    red = np.add(0.5 * red, 0.5 * image)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        output = np.where(mask, red, image).astype(np.uint8)
    else:
        output = image
    return output


def detect_and_visualise(model, image_path=None):
    assert image_path
    
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    
    # Color splash
    output = visualise_mask(image, r['masks'])
    
    output_path = image_path.replace("bmp","jpg").replace("test","test\\results")
    # Save output
    skimage.io.imsave(output_path, output)
    print("Saved to ", output_path)

############################################################
#  COCO Evaluation
############################################################

def build_deepweeds_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "deepweeds"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_deepweeds(model, dataset, deepweeds, eval_type="segm", limit=0, save=False, image_ids=None, subset="test"):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick DeepWeeds images from the dataset
    image_ids = image_ids or dataset.image_ids
    
    # Set directory
    args.dataset_dir = args.dataset_dir + "\\" + subset

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding DeepWeeds image IDs.
    deepweeds_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    custom_metrics = []
    results = []
    ious = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        image_info = deepweeds.loadImgs(deepweeds_image_ids[image_id])
        image_path = args.dataset_dir + "\\" + image_info[0]['file_name']

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        
        # Compute custom metrics
        annotation = dataset.load_mask(image_id)
        prediction = r['masks']
        custom_metrics.append(compute_custom_metrics(annotation[0], prediction))
        
        # Color and save output
        if (save):
            output = visualise_mask(image, r['masks'])
            output_path = image_path.replace("bmp","jpg").replace(subset, subset + "\\results")
            # Save output
            skimage.io.imsave(output_path, output)
            print("Saved to ", output_path)
        

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_deepweeds_results(dataset, deepweeds_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)
        

    # Load results. This modifies results with additional attributes.
    deepweeds_results = deepweeds.loadRes(results)

    # Evaluate detection metrics (Precision and Recall)
    cocoEval = COCOeval(deepweeds, deepweeds_results, eval_type)
    cocoEval.params.imgIds = deepweeds_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Calculate IOUs for each image
    #IoUs = []
    #for i in range(limit):
    #    IoU = cocoEval.computeIoU(i+1, 1)
    #    IoUs.append(IoU)
    #    print(IoU)
    #print("Average IoU = {}".format(np.average(IoUs)))
    
    print_custom_metrics(custom_metrics)
    
    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

def print_custom_metrics(custom_metrics):
    avg_pixel_acc = 0
    avg_mean_pixel_acc = 0
    avg_mean_iou = 0
    avg_freq_iou = 0
    for item in custom_metrics:
        avg_pixel_acc += item['pixel_acc']
        avg_mean_pixel_acc += item['mean_pixel_acc']
        avg_mean_iou += item['mean_iou']
        avg_freq_iou += item['freq_iou']
    avg_pixel_acc /= len(custom_metrics)
    avg_mean_pixel_acc /= len(custom_metrics)
    avg_mean_iou /= len(custom_metrics)
    avg_freq_iou /= len(custom_metrics)
    print(' Average Pixel Accuracy = {:0.5f}'.format(avg_pixel_acc))
    print(' Average Balanced Pixel Accuracy = {:0.5f}'.format(avg_mean_pixel_acc))
    print(' Average IoU = {:0.5f}'.format(avg_mean_iou))
    print(' Average Frequency Weighted IoU = {:0.5f}'.format(avg_freq_iou))
    
def compute_custom_metrics(annotation, prediction):
    
    # Calculate TP, FP, TN, FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    height, width, dimensions = annotation.shape
    for i in range(height):
        for j in range(width):
            if prediction[i][j][0]: # Positive
                if annotation[i][j][0]: # True Positive
                    TP = TP + 1
                else: # False Positive
                    FP = FP + 1
            else: # Negative
                if annotation[i][j][0]: # False Negative
                    FN = FN + 1
                else: # True Negative
                    TN = TN + 1   
    
    # Evaluate metrics
    # Pixel accuracy
    pixel_acc = (TN + TP) / (TP + TN + FP + FN)
    # Mean pixel accuracy
    mean_pixel_acc = (TP / (TP + FN) + TN / (TN + FP)) / 2
    # Mean IoU
    mean_iou = (TP / (TP + FN + FP) + TN / (TN + FN + FP)) / 2
    # Frequency weighted IoU
    freq_iou = ((TP + FN) * TP / (TP + FN + FP) + (TN + FP) * TN / (TN + FN + FP)) / (TP + TN + FP + FN)
    
    return {'pixel_acc':pixel_acc, 'mean_pixel_acc':mean_pixel_acc, 'mean_iou':mean_iou, 'freq_iou':freq_iou}


############################################################
#  MAIN PROGRAM
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect weeds.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset_dir', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the DeepWeeds dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to test the model on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset_dir, "Argument --dataset is required for training"
    args.dataset_dir = os.path.join(ROOT_DIR, args.dataset_dir)
    

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset_dir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DeepWeedsConfig()
    else:
        class InferenceConfig(DeepWeedsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        
        # Evaluation on test dataset
        dataset_test = DeepWeedsDataset()
        deepweeds = dataset_test.load_deepweeds(args.dataset_dir, "test", return_deepweeds=True)
        dataset_test.prepare()
        print("Running DeepWeeds evaluation on test subset.")
        evaluate_deepweeds(model, dataset_test, deepweeds, "segm", limit=20, save=True, subset="test")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
