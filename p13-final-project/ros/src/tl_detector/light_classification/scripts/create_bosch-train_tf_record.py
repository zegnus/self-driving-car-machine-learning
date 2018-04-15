import os
import yaml
import glob
import cv2
import tensorflow as tf
from object_detection.utils import dataset_util 
from tqdm import tqdm

#Define flags for input and output paths
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '.', 'Path to Bosch dataset_train_rgb folder')
FLAGS = flags.FLAGS

def create_tf_example(example):
    
    bosch_train_classes = {"Green" : 1, 
                           "Red" : 2, 
                           "Yellow" : 3, 
                           "off" : 4, 
                           "RedLeft" : 5, 
                           "RedRight" : 6,
                           "GreenLeft" : 7,
                           "GreenRight" : 8,
                           "RedStraight" : 9,
                           "GreenStraight" : 10,
                           "GreenStraightLeft" : 11,
                           "GreenStraightRight" : 12,
                           "RedStraightLeft" : 13,
                           "RedStraightRight" : 14
                          }
    
    height = 720 # Image height
    width = 1280 # Image width
    filename = example['path'].encode() # Filename of the image. Empty if image is not from file
    
    with tf.gfile.GFile(example['path'], 'rb') as fid:
	  #!!!RGB images are being encoded!!!
      encoded_image_data = fid.read() #Encoded image bytes
        
    image_format = b'png' # b'jpeg' or b'png'
  
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    #Bosch box keys are ['occluded', 'label', 'y_max', 'x_max', 'y_min', 'x_min']
    for box in example['boxes']:
        xmins.append(box['x_min']/float(width))
        xmaxs.append(box['x_max']/float(width))
        ymins.append(box['y_min']/float(height))
        ymaxs.append(box['y_max']/float(height))
        classes_text.append(box['label'].encode())
        classes.append(int(bosch_train_classes[box['label']]))
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_example

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  data_folder = FLAGS.input_path
  train_data_folder = os.path.join(data_folder, 'dataset_train_rgb')
  train_annotations_file = os.path.join(train_data_folder, 'train.yaml')
  train_image_folder = os.path.join(train_data_folder, 'rgb', 'train')
  
  print('Reading data from: ', FLAGS.input_path)
  #Load the training dataset YAML file
  train_image_files = glob.glob(os.path.join(train_image_folder,'**','*.png'), recursive=True)
  train_annotations = yaml.load(open(train_annotations_file, 'rb').read())
  #Check that annotations match images
  print('Found {:d} images'.format(len(train_image_files)))
  print('Loaded {:d} annotations'.format(len(train_annotations)))
  assert(len(train_image_files) == len(train_annotations)), 'Number of training annotations does not match training images!'    
  
  #Write the output TFRecord
  print('Writing TFRecord to: ', FLAGS.output_path)    
  n_examples = len(train_annotations)  
  #Annotation keys are 'path' and 'boxes'  
  for example in tqdm(train_annotations, total=n_examples):
    #Fix path for images
    relative_image_path = example['path'][2:].replace('/', os.path.sep)
    example['path'] = os.path.join(train_data_folder, relative_image_path)
	#Create TF example from data
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())
  print('Data written successfully.')
  writer.close()
  
if __name__ == '__main__':
  tf.app.run()