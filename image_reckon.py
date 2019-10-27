import tensorflow as tf
import numpy as np
import Image

imagePath = '/home/samyak/test_images/waterbody_test6.jpg'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'
crop_image_path = '/home/samyak/test_images/cropped_image.jpg'

###################################################################################################

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(crop_image_path):
        tf.logging.fatal('File does not exist %s', crop_image_path)
        return answer

    image_data = tf.gfile.FastGFile(crop_image_path, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer


#####################################################################################################



if __name__ == '__main__':

	#Creates queue of files
	filename_queue = tf.train.string_input_producer([imagePath]) #  list of files to read

	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	resize_h = 1024
	resize_w = 1024
	crop_image_inference = 0
	

	my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

	init_op = tf.initialize_all_variables()
	with tf.Session() as sess:
	  sess.run(init_op)

	  # Start populating the filename queue.

	  coord = tf.train.Coordinator()
	  threads = tf.train.start_queue_runners(coord=coord)


	  for i in range(1): #length of your filename list
	    image = my_img.eval() #here is your image Tensor :) 

	  print(image.shape)

	  resized_image = tf.image.resize_images(image, [resize_w, resize_h])

	  print (resized_image.shape)
	  
	  

	  for j in range(2):
	    for k in range(2):
	      cropped_image_file = open(crop_image_path, "w")
	      crop_image = tf.image.crop_to_bounding_box(resized_image,j * (resize_h/2),k * (resize_w/2) ,resize_h/2 , resize_w/2)
	      crop_image = tf.cast(crop_image,tf.uint8)
	      crop_image_jpeg = tf.image.encode_jpeg(crop_image)
	      cropped_image_file.write(crop_image_jpeg.eval())
	      cropped_image_file.close()
              crop_image_inference = run_inference_on_image() 
	      print crop_image_inference
	      print (crop_image.shape)
	      #run_inference_on_image(crop_image)

	  #Image.fromarray(np.asarray(resized_image)).show()

	  coord.request_stop()
	  coord.join(threads)
