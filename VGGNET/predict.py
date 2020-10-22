#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: Shang
# @Date  : 2020/8/22

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: Shang
# @Date  : 2020/8/8

import os, sys
import numpy as np
import tensorflow as tf
import datetime
#from model import AlexNetModel
from model import VggNetModel
import cv2
from PIL import Image
sys.path.insert(0, '/home/ugrad/Shang/animal/tensorflow-cnn-finetune-master/utils')
from preprocessor import BatchPreprocessor
os.environ["CUDA_VISIBLE_DEVICES"] = "5"



tf.app.flags.DEFINE_float('dropout_keep_prob', 1, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 232, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')

FLAGS = tf.app.flags.FLAGS


def predict(path,modelpath):
    with tf.Graph().as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, [1,224, 224, 3])
        dropout_keep_prob = tf.placeholder(tf.float32)
        imgs = []
        # path='/home/ugrad/Shang/animal/1_.jpg'
        # image = cv2.imread(path,0)

        # cv2.imwrite(path,img)
        img=cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        imgs.append(img)
        # img=Image.open(path)
        # img = np.array(img)
        # img = tf.cast(img, tf.float32)
        # img = tf.reshape(img, [1, 227, 227, 3])

        # Model
        model = VggNetModel(num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_keep_prob)
        logits=model.inference(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Directly restore (your model should be exactly the same with checkpoint)
            # Load the pretrained weights
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, modelpath)
            prediction = sess.run(logits, feed_dict={x: imgs,dropout_keep_prob: 1.})
            # print(prediction)
            max_index = np.argmax(prediction)
            print(max_index)
        return max_index


def writetxt(path,data):
    with open(path,'w',encoding='utf8') as f:
        for i in data:
            f.write(str(i)+'\n')
    f.close()


def main():
    count=0
    err=[]
    imgs=[]
    path='/home/ugrad/Shang/animal/tensorflow-cnn-finetune-master/vggnet/err_yu.txt'
    modelpath='/home/ugrad/Shang/animal/tensorflow-cnn-finetune-master/vggnet/training-all/vggnet_20200831_162842//checkpoint/model_epoch16.ckpt'
    valpath='/home/ugrad/Shang/animal/tensorflow-cnn-finetune-master/data/all/val.txt'
    with open(valpath,'r',encoding='utf8') as f:
        val=f.readlines()
        lines = [i.strip().replace('\n', '') for i in val]
    s = datetime.datetime.now()
    for line in lines:
        i=line.split(' ')
        print(i)
        # img = cv2.imread(i[0])
        # img = cv2.resize(img, (224, 224))
        # img = img.astype(np.float32)
        # imgs.append(img)

        re = predict(i[0], modelpath)

        if re==int(i[1]):
           count += 1
        else:
            err.append(i[0]+' '+str(re))
    e = datetime.datetime.now()
    print((e - s).seconds/len(val))
    print(count,len(val))
    print('acc:',count/len(val))
    print(err)
    # writetxt(path,err)

if __name__ == '__main__':
    main()

