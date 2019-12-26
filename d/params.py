import cv2 as cv
  
L1_LOSS = 1
L2_LOSS = 2


scale = 2
folder_name = '00001_0004'
folder_base_name = '../cnn-3d/3d-images'
# cv.INTER_LINEAR 
# cv.INTER_CUBIC
# cv.INTER_LANCZOS4
# cv.INTER_NEAREST
interpolation_method = cv.INTER_LANCZOS4 
num_epochs = 45
LOSS = L1_LOSS
learning_rate = 1e-3
dim_patch = '_d_14_7_%d' % scale
dim_patch_2 = '_d_2_1_14_7_%d' % scale 
kernel_size = 5 
image_ext = 'png'
folder_data = './data_ckpt/'
layers = 8
num_channels = 1
tf_version = 1.02
ckpt_name = 'model.ckpt'
latest_ckpt_filename = 'latest_epoch_tested'

# for network architecture to be the same everywhere
import networks as nets
network_architecture = nets.SRCNN_late_upscaling_D


def show_params():
	print('\n\n\n\n')
	print('The configuration file is:')
	print('scale = {} '.format(scale))
	print('folder base name = {} '.format(folder_base_name))
	print('folder name = {} '.format(folder_name))
	print('image extension = {} '.format(image_ext))
	print('interpolation method = {} '.format(interpolation_method))
	print('num epochs = {} '.format(num_epochs))
	print('loss = {} '.format(LOSS))
	print('learning rate = {} '.format(learning_rate))
	print('dim patch = {} '.format(dim_patch)) 
	print('kernel size = {} '.format(kernel_size))
	print('folder data size = {} '.format(folder_data)) 
	print('num_channels  = {} '.format(num_channels))
	print('\n\n')