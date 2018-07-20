import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import argparse
import os
import time

parser = argparse.ArgumentParser(description='Inference demo')
parser.add_argument(
    '--weights',
    default='network-snapshot-000641.pkl',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')



def main():
	args = parser.parse_args()
	# Initialize TensorFlow session.
	tf.InteractiveSession()
	# Import official CelebA-HQ networks.
	with open(args.weights, 'rb') as file:
	    G, D, Gs = pickle.load(file)

	 # create folder.
	for i in range(1000):
		path = 'results/generate/try_{}'.format(i)
		if not os.path.exists(path):
			os.system('mkdir -p {}'.format(path))
			break
	randomizer = np.random.RandomState(int(time.time()))

	for i in range(0, 50, 10):
		# Generate latent vectors.
		latents = randomizer.randn(10, *Gs.input_shapes[0][1:])
		# latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

		# Generate dummy labels (not used by the official networks).
		labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

		# Run the generator to produce a set of images.
		images = Gs.run(latents, labels)

		# Convert images to PIL-compatible format.
		images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
		images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

	
		# Save images as PNG.
		for idx in range(images.shape[0]):
			real_idx = i + idx
			print("{}-th image".format(real_idx))
			fname = os.path.join(path, '_gen{}.png'.format(real_idx))
			PIL.Image.fromarray(images[idx], 'RGB').save(fname)


if __name__ == '__main__':
	main()
