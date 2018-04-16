import tensorflow as tf
import sys
import numpy as np

def get_iterator(tfrecord_path):
    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

def reconstruct_tfrecord_rawdata(tfrecord_path):
    debug_flag_lv0=True
    debug_flag_lv1=True
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

    print 'now Reconstruct Image Data please wait a second'

    reconstruct_image=[]
    #caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)
    n=len(list(record_iter))
    record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)

    print 'The Number of Data :' , n
    ret_img_list=[]
    ret_lab_list=[]
    ret_filename_list=[]
    for i, str_record in enumerate(record_iter):
        msg = '\r -progress {0}/{1}'.format(i,n)
        sys.stdout.write(msg)
        sys.stdout.flush()

        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = (example.features.feature['filename'].bytes_list.value[0])
        image = np.fromstring(raw_image , dtype = np.uint8)
        image = image.reshape((height , width , -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_filename_list.append(filename)
    ret_img=np.asarray(ret_img_list)
    ret_lab=np.asarray(ret_lab_list)
    if debug_flag_lv1 == True:
        print ''
        print 'images shape : ',np.shape(ret_img)
        print 'labels shape : ',np.shape(ret_lab)
        print 'length of filenames : ',len(ret_filename_list)
    return ret_img , ret_lab , ret_filename_list

def get_shuffled_batch( tfrecord_path , batch_size , resize ):
    resize_height , resize_width  = resize
    filename_queue = tf.train.string_input_producer([tfrecord_path] , num_epochs=10)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64),
        'filename': tf.FixedLenFeature([] , tf.string)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    filename = tf.cast(features['filename'] , tf.string)

    image_shape = tf.stack([height , width , 3 ])  #image_shape shape is ..
    image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
    image=tf.reshape(image ,  image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=resize_height,
                                           target_width=resize_width)
    images  , labels = tf.train.shuffle_batch([image ,label] , batch_size =batch_size  , capacity =30 ,num_threads=1 , min_after_dequeue=10)
    return images  ,labels

def read_one_example( tfrecord_path , batch_size , resize ):
    filename_queue = tf.train.string_input_producer([tfrecord_path] , num_epochs=10)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    image_shape = tf.pack([height , width , 3 ])
    image=tf.reshape(image ,  image_shape)
    if not resize == None :
        resize_height , resize_width  = resize
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
#    images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image,label