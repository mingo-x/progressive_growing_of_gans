import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description='Inference demo')
parser.add_argument(
    '--weights',
    default='network-snapshot-000641.pkl',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument(
    '--input_dir',
    type=str,
    metavar='PATH')



def main():
	args = parser.parse_args()
	# Initialize TensorFlow session.
	tf.InteractiveSession()
	# Import official CelebA-HQ networks.
	with open(args.weights, 'rb') as file:
	    _, D, _ = pickle.load(file)

	for image_name in os.listdir(args.input_dir):
		image = Image.open(os.path.join(args.input_dir, image_name))
		image_np = np.array(image.getdata(), np.uint8).reshape(1, image.size[1], image.size[0], 3).astype(np.float32)
		image_np = image_np.transpose(0, 3, 1, 2)
		image_np /= 255.
		image_np *= 2.
		image_np -= 1.
		image_np = np.resize(image_np, (1, 3, image.size[1], 1, image.size[0], 1))
		image_np = np.tile(image_np, (1, 1, 1, 2, 1, 2))
		image_np = np.resize(image_np, (1, 3, image.size[1]*2, image.size[0]*2))
#		print(D.input_shapes)
		# Generate dummy labels (not used by the official networks).
#		label = np.zeros([image_np.shape[0]] + D.input_shapes[1][1:])

		score = D.run(image_np)
		print(score[0][0][0])

if __name__ == '__main__':
	main()
