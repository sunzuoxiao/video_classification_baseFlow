#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

#path :视频源文件地址，intervalF隔几帧取一次光流，W,H：resize后的宽高，timeMax :最高取多少帧,beignF:从视频的第几帧开始
def extract_flow_stream(path, intervalF, W, H, beginFrame,timeMax):
    C = 0
    time = 0
    beginF = 0
    stackFlow = []
    preFrameGray = []
    originImage = []
    cap = cv2.VideoCapture(path)
    isOk, preFrame = cap.read()
    flow = np.zeros([W, H, 2], np.float32)
    while isOk and time<timeMax:
        isOk, nextFrame = cap.read()
        if beginF<beginFrame:
            beginF = beginF + 1
        elif beginFrame == beginF:
            preFrameBGR = cv2.resize(nextFrame, (W, H), interpolation=cv2.INTER_LINEAR)
            preFrameRGB = cv2.cvtColor(preFrameBGR, cv2.COLOR_BGR2RGB)
            preFrameGray = cv2.cvtColor(preFrameRGB, cv2.COLOR_RGB2GRAY)
            originImage = preFrameRGB
            beginF = beginF + 1
        else:
            if (C % intervalF == 0):
                nextFrameBGR = cv2.resize(nextFrame, (W, H), interpolation=cv2.INTER_LINEAR)
                nextFrameRGB = cv2.cvtColor(nextFrameBGR, cv2.COLOR_BGR2RGB)
                nextFrameGray = cv2.cvtColor(nextFrameRGB, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(preFrameGray, nextFrameGray, flow, 0.5, 5, 15, 3, 5, 1,
                                                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                flow = flow.astype(np.int32)
                print flow
                preFrameGray = nextFrameGray
                if time == 0:
                    stackFlow = flow
                else:
                    stackFlow = np.concatenate((stackFlow, flow), -1)
                time = time + 1
    return originImage, stackFlow

def creatTFRecord(originImage, stackFlow, label, out_file, path_name):
    out_file_path = os.path.join(out_file,'video_flow.TFRecord')
    writer = tf.python_io.TFRecordWriter(out_file_path)
    #按照实际情况制定for循环
    path_file_list = os.listdir(path_name)
    for single_file_path in path_file_list:
        # 根据情况拼接视频文件名字,  待补代码,以及extract_flow_stream的参数
        originImage_raw, stackFlow_raw = extract_flow_stream()
        originImage_raw = originImage_raw.tobytes()
        stackFlow_raw = stackFlow_raw.tobytes()
        label = single_file_path
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'originImage_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[originImage_raw])),
            'stackFlow_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stackFlow_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()



def read_decodeTFTecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'originImage':tf.FixedLenFeature([], tf.string),
                                           'stackFlow':tf.FixedLenFeature([], tf.string),
                                           'label':tf.FixedLenFeature([], tf.int32)
                                       })
    originImage_raw = tf.decode_raw(features['originImage'], tf.uint8)
    originImage_raw = tf.reshape(originImage_raw, [224, 224, 3])
    stactFlow = tf.decode_raw(features['stackFlow'], tf.uint8)
    stactFlow = tf.reshape(stactFlow, [224, 224, 20])
    label = tf.cast(features['label'], tf.int32)

    return  originImage_raw, stactFlow, label

def test_read(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'data_raw' : tf.FixedLenFeature([], tf.int64),
                                       })

    data_raw = tf.cast(features['data_raw'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    return data_raw, label
def test():
    writer = tf.python_io.TFRecordWriter("/Users/szx/Desktop/train.tfrecords")
    for i in range(100):
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'data_raw': tf.train.Feature(int64_list=tf.train.Int64List(value=[i*100]))
            }))
        writer.write(example.SerializeToString())



if __name__ == '__main__':

   # extract_flow_stream(path, intervalF, W, H, timeMax)
   tt, xx = extract_flow_stream(path='/Users/szx/Desktop/504551105_old.mp4', intervalF=1, W=224, H=224, beginFrame=10,timeMax=20)
   # print tt
   # print 'ddddddddddd'
   # print xx
   # print np.max(xx,0)
   # print np.shape(tt)
   # print np.shape(xx)


