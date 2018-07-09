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
    default='karras2018iclr-celebahq-1024x1024.pkl',
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
		path = 'repo/generate/try_{}'.format(i)
		if not os.path.exists(path):
			os.system('mkdir -p {}'.format(path))
			break

	for i in range(50):
		# Generate latent vectors.
		latents = np.random.RandomState(int(time.time())).randn(1, *Gs.input_shapes[0][1:]) # 1000 random latents
		# latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

		# Generate dummy labels (not used by the official networks).
		labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

		# Run the generator to produce a set of images.
		images = Gs.run(latents, labels)

		# Convert images to PIL-compatible format.
		images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
		images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

	
		# Save images as PNG.
		print("{}-th image".format(i))
		fname = os.path.join(path, '_gen{}.png'.format(i))
		PIL.Image.fromarray(images[0], 'RGB').save(fname)


if __name__ == '__main__':
	main()
