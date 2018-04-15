import os
import yaml
import glob
import cv2
import random
import tensorflow as tf
from object_detection.utils import dataset_util 
from tqdm import tqdm

#Define flags for input and output paths
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '.', 'Path to Bosch dataset_test_rgb folder')
flags.DEFINE_integer('eval_samples', 500, 'Number of evaluation samples to use')
FLAGS = flags.FLAGS

def create_tf_example(example):
    
    bosch_test_classes = {"Green" : 1, 
                           "Red" : 2, 
                           "Yellow" : 3, 
                           "off" : 4, 
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
        classes.append(int(bosch_test_classes[box['label']]))
    
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
  n_samples = FLAGS.eval_samples
  test_data_folder = os.path.join(data_folder, 'dataset_test_rgb')
  test_annotations_file = os.path.join(test_data_folder, 'test.yaml')
  test_image_folder = os.path.join(test_data_folder, 'rgb', 'test')
  
  print('Reading data from: ', FLAGS.input_path)
  #Load the test dataset YAML file
  test_image_files = glob.glob(os.path.join(test_image_folder,'**','*.png'), recursive=True)
  test_annotations = yaml.load(open(test_annotations_file, 'rb').read())
  #Check that annotations match images
  print('Found {:d} images'.format(len(test_image_files)))
  print('Loaded {:d} annotations'.format(len(test_annotations)))
  assert(len(test_image_files) == len(test_annotations)), 'Number of test annotations does not match training images!'    
  examples = random.sample(test_annotations, n_samples)
  print('Selected {:d} random samples'.format(len(examples)))
  #Write the output TFRecord
  print('Writing TFRecord to: ', FLAGS.output_path)    
  #Annotation keys are 'path' and 'boxes'  
  for example in tqdm(examples, total=n_samples):
    #Fix path for images
    filename = example['path'].split('/')[-1]
    example['path'] = os.path.join(test_image_folder, filename)
	#Create TF example from data
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())
  print('Data written successfully.')
  writer.close()
  
if __name__ == '__main__':
  tf.app.run()