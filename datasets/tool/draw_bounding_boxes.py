import numpy as np
import os
import re 
import argparse
from PIL import Image
import scipy
from scipy.signal import *


def parse_args():
    parse = argparse.ArgumentParser(description='Draw ground truth bounding boxes.')
    parse.add_argument('--annots_dir', dest='annots_dir', help='Annotations directory')
    parse.add_argument('--images_dir', dest='images_dir', help='Images directory')
    parse.add_argument('--output_dir', dest='output_dir', help='Image output directory')

    args = parse.parse_args()
    return args


def load_gt_bbox(filepath):
    with open(filepath) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+ \d+', data)
    
    nums_obj = len(objs)
    gtBBs = np.zeros((nums_obj, 4))
    for idx, obj in enumerate(objs):
        info = re.findall('\d+', obj)
        x1 = float(info[0])
        y1 = float(info[1])
        x2 = float(info[2])
        y2 = float(info[3])
        gtBBs[idx, :] = [x1, y1, x2, y2]
    return gtBBs


def draw(bb, im, color):
    L = int(bb[0]); T = int(bb[1]); R = int(bb[2]); B = int(bb[3])

    # draw horizontal line
    for x in xrange(L, R+1):
        for channel in xrange(len(color)):
            im[channel][x][T] = color[channel] # top line
            im[channel][x][B] = color[channel] # bottom line
    
    # draw vertical line
    for y in xrange(T, B+1):
        for channel in xrange(len(color)):
            im[channel][L][y] = color[channel] # left line
            im[channel][R][y] = color[channel] # right line

    return im


def main():
    args = parse_args()

    for image_filename in os.listdir(args.images_dir):
        base_filename = (image_filename.strip().split('.'))[0]

        # load image
        im = Image.open(args.images_dir + '/' + image_filename).convert('RGB')
        im = np.array(im).T
        im_copy = np.copy(im)

        # load ground truth bounding box
        annot_filename = base_filename + '.txt'
        gtBBs = load_gt_bbox(args.annots_dir + '/' + annot_filename)

        # draw ground truth annotations on the image
        for bb in gtBBs:
            draw(bb, im_copy, color=[0,255,0]) # color: green
            
        # output the result image
        output_filepath = args.output_dir + '/' + base_filename + '_gt_bbox.jpg'
        scipy.misc.imsave(output_filepath, im_copy.T)
        print 'Save: ' + output_filepath


if __name__=='__main__':
    main()
