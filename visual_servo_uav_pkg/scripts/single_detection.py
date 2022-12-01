
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import shutil
from visual_servo_uav_msgs.msg import DroneCommand

import rospy 
from sensor_msgs.msg import Image as CompressedImage

import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw
from cv_bridge import CvBridge
import cv_bridge
import numpy as np

import torch

import misc as utils

from detr import build
from drone import make_Drone_transforms

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="drone")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/home/freddyboy/ros_ws/src/visual_servo_uav/dataset/output',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    duration = 0 
  
    filename = os.path.basename(images_path[0])
   
    print("processing...{}".format(filename))
    im = cv2.imread(images_path[0])
    im = cv2.resize(im, (416,416), interpolation = cv2.INTER_AREA)
    print(im.size)
    orig_image = Image.open(images_path[0])
    orig_image=orig_image.resize((416, 416))
    w, h = orig_image.size
    print(w,h)
    size_w=416
    size_h=416
    x_d=w/2
    y_d=h/2

    

    A_d=5   #Dobbiamo acquisirla
    print("desired values",x_d,y_d,A_d)
    transform = make_Drone_transforms("val")
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    image, targets = transform(orig_image, dummy_target)
    image = image.unsqueeze(0)
    image = image.to(device)


    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)

        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])

        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1])

        ),

    ]

    start_t = time.perf_counter()
    outputs = model(image)
    end_t = time.perf_counter()
    
    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
   
    #print(outputs["pred_logits"])
    pred_logits=outputs['pred_logits'][0][:, :]
    #print(pred_logits)
    pred_boxes=outputs['pred_boxes'][0]
    print( "pred_boxes val:\n " , pred_boxes)
    max_output = pred_logits.softmax(-1).max(-1)
    print ( "max_output val:\n ", max_output )
    topk = max_output.values.topk(1)
    print( "top_k val:\n" , topk )
    pred_logits = pred_logits[topk.indices]
    # print("pred_logits val:\n ", pred_logits)
    
    pred_boxes = pred_boxes[topk.indices] 

    x, y, w, h = pred_boxes.numpy()[0] 

    logit = pred_logits.numpy()[0][-1]

    OR = np.exp(logit)
    conf = OR/(1+OR)
    #print(conf)
    infer_time = end_t - start_t
    duration += infer_time
    print("Processing...{} ({:.3f}s)".format(filename, infer_time))
    
    im = cv2.rectangle(im,(int((x-w/2)*size_w),int((y+h/2)*size_w)),(int((x+w/2)*size_h),int((y-h/2)*size_h)),(0,0,255),2)
 
   
    cv2.putText(im,'Drone: '+str(round(conf,4)),(int((x-w/2)*416),int((y-h/2)*416)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
    
    filesave = '/home/freddyboy/ros_ws/src/visual_servo_uav/dataset/output'+filename
    print("#########")
    cv2.imwrite(filesave,im)        
    avg_duration = duration / (len(images_path))
    print("Avg. Time: {:.3f}s".format(avg_duration))
    x, y, w, h = pred_boxes.numpy()[0]
    x_1 = int( ( x-w/2 )*416 ) 
    x_2 = int( ( x+w/2 )*416 )
    y_1 = int( ( y-h/2 )*416 )
    y_2 = int( ( y+h/2 )*416 )
    x_t = np.min( ( x_1 , x_2 ) ) + np.abs( x_2 - x_1 )/2
    y_t = np.min( ( y_1 , y_2 ) ) + np.abs( y_2 - y_1 )/2
    A_t = np.abs( x_2 - x_1 )*np.abs( y_2 - y_1 )
    
    print("desired area for calibration:",A_t)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, _, postprocessors = build(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    shutil.rmtree('/home/freddyboy/ros_ws/src/visual_servo_uav/dataset/output')
    os.mkdir('/home/freddyboy/ros_ws/src/visual_servo_uav/dataset/output')
    image_paths = get_images(args.data_path)
    infer(image_paths, model, postprocessors, device, args.output_dir)