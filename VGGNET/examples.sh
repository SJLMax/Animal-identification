#!/bin/sh
# 3/4/5
python finetune.py --learning_rate "0.00001" --num_epochs 20 --num_classes 15 --train_layers "fc8,fc7,fc6,conv5_3,conv5_2,conv5_1,conv4_3,conv4_2,conv4_1,conv3_3,conv3_2,conv3_1,conv2_2,conv2_1,conv1_2,conv1_1"

# 5
python finetune.py --learning_rate "0.00001" --num_epochs 20 --num_classes 9 --train_layers "fc8,fc7,fc6,conv5_3,conv5_2,conv5_1"
--multi_scale "225,256"