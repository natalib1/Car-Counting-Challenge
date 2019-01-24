import sys, os, caffe, cv2
import _init_paths
import numpy as np
import argparse
import re

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

CLASSES = ('__background__', 'car')
CONF_THRESH = 0.5 # choose score confidence > 0.5
NMS_THRESH = 0.3  # filter the box overlap > 0.3 with each other.

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='evaluate MSE and RMSE')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', dest='prototxt', help='file path to *.prototxt')
    parser.add_argument('--caffemodel', dest='caffemodel', help='file path to *.caffemodel')
    parser.add_argument('--test_images_dir', dest='test_images_dir', help='The testing images directory')
    parser.add_argument('--num_proposals', dest='num_proposals', help='number of proposals to use')
    parser.add_argument('--annots_dir', dest='annots_dir', help='annotations directory')

    args = parser.parse_args()
    return args


def get_number_groundTruth(annots_dir, image_name):
    filename = (image_name.strip().split('.'))[0] + '.txt'
    with open(annots_dir + '/' + filename) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+ \d+', data)
    return len(objs)


def forward_image(net, input_dir, image_name):
    
    # Load the demo image
    filepath = input_dir + '/' + image_name
    if not os.path.exists(filepath):
        raise IOError('Image at' + ' \"' + str(filepath) + '\" '  + 'is not found. Please check whether the file or directory exists or not.!\n')
    im_file = image_name
    im = cv2.imread(filepath)

    # get the number of bounding boxes
    scores, boxes = im_detect(net, im)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # skipped background class 
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        num_det = len(inds)
   
    return num_det


if __name__=='__main__':
    args = parse_args()
    
    # config
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = args.num_proposals
    prototxt = args.prototxt
    caffemodel = args.caffemodel

    # load model by caffe
    if not os.path.isfile(caffemodel):
        raise IOError('{:s} not found. Please check whether the file path exists or not.'.format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}.'.format(caffemodel)

    # load image filenames
    im_names = []
    for image_filename in os.listdir(args.test_images_dir):
        im_names.append(image_filename)
    len_testing_images = len(im_names)

    # evaluate on each input testing image
    absolute_error = np.zeros(len_testing_images)
    square_error = np.zeros(len_testing_images)
    index = 0
    for im_name in im_names:
        # ground truth object number
        num_gt = get_number_groundTruth(args.annots_dir, im_name)
        
        # detect predicted object number 
        print '------'
        print 'Forward the image: {}'.format(args.annots_dir, im_name)
        num_det = forward_image(net, args.test_images_dir, im_name)

        # record the error
        absolute_error[index] = np.abs(num_det - num_gt)
        square_error[index] = (num_det - num_gt) ** 2
        info = '{}: #Count@NMS{}@CONF{} = {}, #GT = {}, Difference = {}, sum_of_AE = {}, MAE = {}'.format(str(index), str(NMS_THRESH), str(CONF_THRESH), str(num_det), str(num_gt), str(absolute_error[index]), str(np.sum(absolute_error)), str(np.mean(absolute_error)))
        print info
        index += 1

    # Final MAE and RMSE
    print '\n================================'
    MAE = np.mean(absolute_error)
    print 'Mean Absolute Error (MAE) = ', str(MAE)
    RMSE = np.sqrt(np.mean(square_error))
    print 'Root Mean Square Error (RMSE) = {}\n'.format(str(RMSE))


