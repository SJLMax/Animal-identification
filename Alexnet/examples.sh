#!/bin/sh
--multi_scale "228,256"
python finetune.py --num_classes 232 --num_epochs 20 --train_layers "fc8,fc7,fc6,conv5"
python finetune.py --num_classes 9 --num_epochs 20 --train_layers "fc8,fc7,fc6"



python finetune.py --num_classes 232 --num_epochs 20 --train_layers "fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1"
# full training
