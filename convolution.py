from PIL import Image
import numpy as np
output_matrix = []

#return the result of convolution with a kernel
def single_pixel_convolution(input_image, start_x, start_y, kernel, kernel_height, kernel_width):
    output_red = 0
    output_green = 0
    output_blue = 0
    for i in range (0,kernel_height):
        for j in range (0,kernel_width):
            output_red = output_red + input_image[start_x+i][start_y+j][0]*kernel[i][j]
            output_green = output_green + input_image[start_x+i][start_y+j][1]*kernel[i][j]
            output_blue = output_blue + input_image[start_x+i][start_y+j][2]*kernel[i][j]
    return (output_red,output_green,output_blue)

#convolution with output dimensions lesser than the input image but with less distortions 
def convolution_2D(input_image, image_width, image_height, kernel, kernel_height, kernel_width):
    reduced_width = image_width-kernel_width+1
    reduced_height = image_height-kernel_height+1
    global output_matrix
    output_matrix = np.zeros(shape=(reduced_height,reduced_width,3))
    for i in range (0,reduced_height):
        for j in range (0,reduced_width):
            output_matrix[i][j] = single_pixel_convolution(input_image, i, j, kernel, kernel_height, kernel_width)
            if (output_matrix[i][j][0]>255):
                output_matrix[i][j][0] = 255
            elif (output_matrix[i][j][0]<0):
                output_matrix[i][j][0] = 0
            if (output_matrix[i][j][1]>255):
                output_matrix[i][j][1] = 255
            elif (output_matrix[i][j][1]<0):
                output_matrix[i][j][1] = 0
            if (output_matrix[i][j][2]>255):
                output_matrix[i][j][2] = 255
            elif (output_matrix[i][j][2]<0):
                output_matrix[i][j][2] = 0            

#convert numpy array to RGB array so that it can be saved in the form of an image
#def convertNumpytoRGBArray():

input_image_matrix = np.asarray(Image.open("source.jpg").convert("RGB"))

#print (input_image_matrix)
#print (input_image_matrix.shape)
input_image_height = len(input_image_matrix)
input_image_width = len(input_image_matrix[0])

#defining kernel for convolution
convolution_kernel = np.array([[0,0,0,0,0], [0,0,-1,0,0], [0,-1,5,-1,0], [0,0,-1,0,0], [0,0,0,0,0]])
convolution_kernel_width = 5
convolution_kernel_height = 5
convolution_2D(input_image_matrix, input_image_width, input_image_height, convolution_kernel, convolution_kernel_height, convolution_kernel_width)
output_image = Image.fromarray(output_matrix, 'RGB')
output_image.show()
