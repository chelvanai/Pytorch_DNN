from nbformat import read
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL


# Object detection dataset json read with image
def read_json_dataset(json_file, image_dir):
    import json
    import os
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    # read json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # read image
    image_path = os.path.join(image_dir, data['imagePath'])
    image = Image.open(image_path)
    image = TF.to_tensor(image)
    image = image.unsqueeze(0)

    # read bounding box
    bbox = np.array(data['boundingBox'])
    bbox = bbox.reshape(1, 4)
    bbox = torch.from_numpy(bbox)

    # read label
    label = np.array(data['label'])
    label = label.reshape(1, 1)
    label = torch.from_numpy(label)

    # read image and bounding box and label
    return image, bbox, label

read_json_dataset('/home/jhkim/Desktop/Convolution_NN/data/train/train_data.json', '/home/jhkim/Desktop/Convolution_NN/data/train/')

# ROI pooling function
def roi_pooling(input, rois, output_size):
    # input: (N, C, H, W)
    # rois: (N, 4)
    # output_size: (H, W)
    batch_size, _, input_size_h, input_size_w = input.size()
    num_rois = rois.size(0)

    # output: (N, C, H, W)
    output = torch.zeros(batch_size, 256, output_size[0], output_size[1])

    # roi_pooling
    for i in range(num_rois):
        roi = rois[i]
        batch_idx = roi[0].item()
        roi_h = roi[2].item()
        roi_w = roi[3].item()
        roi_h_start = int(round(roi[1].item()))
        roi_w_start = int(round(roi[1].item()))

        # roi_pooling
        roi_pool = input[batch_idx, :, roi_h_start:roi_h_start+output_size[0], roi_w_start:roi_w_start+output_size[1]]
        output[batch_idx, :, :, :] += roi_pool

    return output

Resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

FasterRCNN = nn.Sequential(
    Resnet,
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

