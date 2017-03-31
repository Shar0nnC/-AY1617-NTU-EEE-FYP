import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from collections import namedtuple
from showasImage import *
import colorsys

def color16(input_image):
	
	[rows,cols,_]=input_image.shape
	
	# Input image is a numpy 3D array
	ColorChannel = namedtuple("ColorChannel", "c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16")
	
	red = input_image[:,:,2]
	green = input_image[:,:,1]
	blue = input_image[:,:,0]
	
	
	
	red = red.astype(float)
	green = green.astype(float)
	blue = blue.astype(float)
	
	
	# RGB images
	red_image = red
	green_image = green
	blue_image = blue 
	
	
	#red = red.flatten()
	#blue = blue.flatten()
	#green = green.flatten()
	

	
	
	# BR image
	br_num = blue - red
	br_den = blue + red
	BR = br_num/br_den
	BR[np.isnan(BR)] = 0
	BR_image = showasImage(BR) 
	
	
	# HSV image
	H_image = np.zeros([rows,cols])
	S_image = np.zeros([rows,cols])
	V_image = np.zeros([rows,cols])
	
	# YIQ image
	Y_image = np.zeros([rows,cols])
	I_image = np.zeros([rows,cols])
	Q_image = np.zeros([rows,cols])
	
	# Chroma channels
	C_image = np.zeros([rows,cols])
	
	for i in range(rows):
		for j in range(cols):
			r_value = red[i,j]
			g_value = green[i,j]
			b_value = blue[i,j]
			
			hsv_tuple = colorsys.rgb_to_hsv(r_value,g_value,b_value)
			
			H_image[i,j] = hsv_tuple[0]
			S_image[i,j] = hsv_tuple[1]
			V_image[i,j] = hsv_tuple[2]
			
			yiq_tuple = colorsys.rgb_to_yiq(r_value,g_value,b_value)

			Y_image[i,j] = yiq_tuple[0]
			I_image[i,j] = yiq_tuple[1]
			Q_image[i,j] = yiq_tuple[2]
			
			rgb_arr = np.asarray([r_value,g_value,b_value])
			C_image[i,j] = np.max(rgb_arr) - np.min(rgb_arr)
			
			
	H_image = showasImage(H_image)
	S_image = showasImage(S_image)
	V_image = showasImage(V_image)
	
	Y_image = showasImage(Y_image)
	I_image = showasImage(I_image)
	Q_image = showasImage(Q_image)
	
	C_image = showasImage(C_image)
	

	
	

	
	# Other ratio images
	thirteen =  red/blue
	thirteen[np.isnan(thirteen)] = 0
	thirteen = showasImage(thirteen)
	
	fourteen = red - blue
	fourteen = showasImage(fourteen)
	
	#Lab color space
	L_image = 0.2126*red + 0.7152*green + 0.0722*blue
	

	cc = ColorChannel(c1 = red_image, c2 = green_image, c3 = blue_image ,c4 = H_image, c5 = S_image, c6 = V_image ,c7 = Y_image, c8 = I_image, c9 = Q_image, c10 = L_image, c11 = "bar", c12 = "baz",c13 = thirteen ,c14 = fourteen, c15 = BR_image, c16 = C_image)
	
	return(cc)
