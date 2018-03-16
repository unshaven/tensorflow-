'''
迁移学习(Transfer learning) 顾名思义就是就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。
考虑到大部分数据或任务是存在相关性的，所以通过迁移学习我们可以将已经学到的模型参数（也可理解为模型学到的知识）
通过某种方式来分享给新模型从而加快并习（starting from scratch，tabula rasa）。
优化模型的学习效率不用像大多数网络那样从零学
'''

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#Inception模型瓶颈层模型节点个数
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = '../Inception_v3'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = '../tmp/bottleneck'
INPUT_DATA = 'flower_photos'
#验证数据百分比
VALIDATION_PERCENTAGE = 10
#测试数据百分比
TEST_PERCENTAGE = 10
#定义神经网路的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100
#这个函数从数据文件夹中读取所有的照片列表,并且按照训练,验证,测试数据分开
def create_image_lists(testing_percentage,validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #得到的第一个目录是当前目录,不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        #获取当前目录下所有的有效图片文件
        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
        #初始化当前类别的数据集,训练集,和验证集
        label_name = dir_name.lower()
        training_images = []
        validation_images = []
        testing_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        #将当前类别的数据放入字典
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    return result
def get_image_path(image_lists,image_dir,label_name,index,category):
    #获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    #该图片属于什么数据集
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    #获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path
def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    #用训练好的Inception_v3模型处理一张图片
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    #经过卷积神经网络的结果是一个四维的向量,需要压缩成一个一维数组
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    if not os.path.exists(bottleneck_path):
        #获取原始图片的路径
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        image_data = gfile.FastGFile(image_path,'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        #直接从文件中获取图片相应的特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values
#这个函数随机获取一个batch的图片作为训练数据
#train_bottleneck,train_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,
                                  jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        #随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]  #?
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,
                                              jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths
#这个函数获取全部的测试数据

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    print("image_lists=  ",image_lists)
    print("label_name_list  ",label_name_list)
    #枚举所有所有类别中和每个类别中的 测试数据
    for label_index,label_name in enumerate(label_name_list):
        category = 'testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths
def main(_):
    #读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=
                                                             [BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    bottleneck_input = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
    #定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')
    #定义一层全连接层来解决新的图片分类问题,因为训练好的Inception_v3已经将原始复杂的图片抽象成简单的容易分类的特征向量
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights)+biases
        final_tensor = tf.nn.softmax(logits)
    #定义交叉损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ground_truth_input,1),logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    #计算准确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #训练过程
        for i in range(STEPS):
            #每次获取一个batch的数据
            train_bottleneck,train_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,feed_dict={bottleneck_input:train_bottleneck,ground_truth_input:train_ground_truth})
            #在验证数据上测试正确率
            if (i % 100) == 0 or (i + 1) == STEPS:
                vadilation_bottlenecks,validation_ground_truth = get_random_cached_bottlenecks(sess,
                n_classes,image_lists,BATCH,'validation',jpeg_data_tensor,bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:vadilation_bottlenecks,ground_truth_input:validation_ground_truth})
                print('Step %d: Validation accuracy is %.1f%%' % (i,validation_accuracy*100))
            #在最后的测试数据上测试正确率
        test_bottlenecks,test_ground_truth = get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy*100))
if __name__ =='__main__':
    tf.app.run()










