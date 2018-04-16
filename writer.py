import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
from PIL import Image
import os ,sys
import tensorflow as tf



def make_tfrecord_rawdata(tfrecord_path , paths , labels):
    """
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param paths: e.g)[./pic1.png , ./pic2.png]
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    debug_flag_lv0=True
    debug_flag_lv1=True
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class : tfrecord_batch | make_tfrecord_rawdata'

    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    paths_labels=zip(paths ,labels)
    error_file_paths=[]
    for ind, (path , label) in enumerate(paths_labels):
        try:
            msg = '\r-Progress : {0}'.format(str(ind) +'/'+str(len(paths_labels)))
            sys.stdout.write(msg)
            sys.stdout.flush()

            np_img=np.asarray(Image.open(path)).astype(np.int8)
            height = np_img.shape[0]
            width = np_img.shape[1]
            raw_img = np_img.tostring()
            dirpath , filename=os.path.split(path)
            filename , extension=os.path.splitext(filename)
            if __debug__ == debug_flag_lv1:
                print ''
                print 'image min', np.min(np_img)
                print 'image max', np.max(np_img)
                print 'image shape' , np.shape(np_img)
                print 'heigth , width',height , width
                print 'filename' , filename
                print 'extension ,',extension


            example = tf.train.Example(features = tf.train.Features(feature = {
                        'height': _int64_feature(height),
                        'width' : _int64_feature(width),
                        'raw_image' : _bytes_feature(raw_img),
                        'label' : _int64_feature(label),
                        'filename':_bytes_feature(tf.compat.as_bytes(filename))
                        }))
            writer.write(example.SerializeToString())
        except IndexError as ie :
            print path
            continue
        except IOError as ioe:
            print path
            continue
        except Exception as e:
            print path
            print str(e)
            continue
    writer.close()
