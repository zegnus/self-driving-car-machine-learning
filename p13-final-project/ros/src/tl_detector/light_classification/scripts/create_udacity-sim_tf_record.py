import os
import yaml
import glob
import cv2
import tensorflow as tf
import numpy as np
import copy
from PIL import Image
from object_detection.utils import dataset_util 
from tqdm import tqdm


#Define flags for input and output paths
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '.', 'Location of data folder')
flags.DEFINE_integer('augment_factor', 5, 'Number of random jitters to apply to Green & Yellow images')
FLAGS = flags.FLAGS

def random_jitter_image(img, boxes, xlate_range):
    img_np = load_image_into_numpy_array(img)
    new_img = np.copy(img_np)
    
    height, width, channels = img_np.shape    
    xlate_x = np.random.uniform(xlate_range) - xlate_range/2
    xlate_y = np.random.uniform(xlate_range) - xlate_range/2
    M_xlate = np.float32([[1,0,xlate_x],[0,1,xlate_y]])
    #Transform image
    img_np = cv2.warpAffine(img_np, M_xlate, (width, height))
    #Transform boxes
    for box in boxes:
        box['xmin'] = box['xmin'] + xlate_x/width
        box['ymin'] = box['ymin'] + xlate_y/height 
        box['xmax'] = box['xmax'] + xlate_x/width 
        box['ymax'] = box['ymax'] + xlate_y/height
        
    return img_np, boxes

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)	

def create_tf_example(example):
    
    train_classes = {"Green" : 1, 
                           "Red" : 2, 
                           "Yellow" : 3, 
                           "off" : 4 
                          }
    
    height = 600 # Image height
    width = 800 # Image width
    filename = example['filename'].encode() # Filename of the image. Empty if image is not from file
    
    with tf.gfile.GFile(example['filename'], 'rb') as fid:
	  #!!!RGB images are being encoded!!!
      encoded_image_data = fid.read() #Encoded image bytes
        
    image_format = b'jpeg' # b'jpeg' or b'png'
  
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    #Bosch box keys are ['occluded', 'label', 'y_max', 'x_max', 'y_min', 'x_min']
    for box in example['annotations']:
        xmins.append(box['xmin'])
        xmaxs.append(box['xmax'])
        ymins.append(box['ymin'])
        ymaxs.append(box['ymax'])
        classes_text.append(box['class'].encode())
        classes.append(int(train_classes[box['class']]))
    
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
	aug_factor = FLAGS.augment_factor
	train_data_folder = os.path.join(data_folder, 'data', 'simulator')
	train_annotations_file = os.path.join(train_data_folder, 'final_annotations.yaml')

	print('Reading data from: ', FLAGS.input_path)
	#Load the training dataset YAML file
	train_annotations = yaml.load(open(train_annotations_file, 'rb').read())
	print('Loaded {:d} annotations'.format(len(train_annotations)))

	#Write the output TFRecord
	print('Writing TFRecord to: ', FLAGS.output_path)    
	n_examples = len(train_annotations)  
	#Example keys are 'filename' and 'annotations'  
	for example in tqdm(train_annotations, total=n_examples):
		filename = example['filename'].split('/')[-1]
		relative_path = example['filename'].replace('/', os.path.sep)
		example['filename'] = os.path.join(data_folder, relative_path)
		if example['annotations']:
			#Augment all the green and yellow images
			if ((example['annotations'][0]['class'] =='Green') | (example['annotations'][0]['class'] =='Yellow')):
				#Create TF example from the original data
				tf_example = create_tf_example(example)
				writer.write(tf_example.SerializeToString())
				#Create examples from the augmented data
		
				for i in range(aug_factor):
					new_sample = {}
					new_path = os.path.join(train_data_folder, 'augmented', example['filename'].split(os.path.sep)[-1].split('.')[0]+'_'+str(i)+'.jpg')
					boxes = copy.deepcopy(example['annotations'])
					image = Image.open(example['filename'])
					image_jitter, boxes_jitter = random_jitter_image(image, boxes, 20)
					new_sample['annotations'] = boxes_jitter
					new_sample['filename'] = new_path
					#print('Writing augmented image to path: ', new_path)
					cv2.imwrite(new_path, cv2.cvtColor(image_jitter, cv2.COLOR_RGB2BGR)) 
					tf_example = create_tf_example(new_sample)
					writer.write(tf_example.SerializeToString())

			#Don't augment red images
			
			else:
				tf_example = create_tf_example(example)
				writer.write(tf_example.SerializeToString())
		else:
			#print('Skipping empty sample')
			tf_example = create_tf_example(example)
			writer.write(tf_example.SerializeToString())
			
	print('Data written successfully.')
	writer.close()
  
if __name__ == '__main__':
  tf.app.run()