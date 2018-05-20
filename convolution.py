from PIL import Image
import numpy as np
output_matrix = []

#return the result of convolution with a kernel
def single_pixel_convolution(input_image, start_x, start_y, kernel, kernel_height, kernel_width):
    output = 0
    for i in range (0,kernel_height):
        for j in range (0,kernel_width):
            output = output + input_image[start_x+i][start_y+j]*kernel[i][j]
    return output

#convolution with output dimensions lesser than the input image but with less distortions 
def convolution_2D(input_image, image_width, image_height, kernel, kernel_height, kernel_width):
    reduced_width = image_width-kernel_width+1
    reduced_height = image_height-kernel_height+1
    global output_matrix
    output_matrix = np.zeros(shape=(reduced_height,reduced_width))
    for i in range (0,reduced_height):
        for j in range (0,reduced_width):
            output_matrix[i][j] = single_pixel_convolution(input_image, i, j, kernel, kernel_height, kernel_width)
            if (output_matrix[i][j]>255):
                output_matrix[i][j] = 255
            elif (output_matrix[i][j]<0):
                output_matrix[i][j] = 0

#produce wrapping effect
def convolution2D_for_wrap_effect(input_image, image_width, image_height, kernel, kernel_height, kernel_width):
    reduced_width = image_width-kernel_width+1
    reduced_height = image_height-kernel_height+1
    global output_matrix
    output_matrix = np.zeros(shape=(reduced_height,reduced_width))
    start_index = int(0.35*reduced_height)
    wrap_index = reduced_height-start_index
    for i in range (0,reduced_height-start_index):
        for j in range (0,reduced_width):
            output_matrix[i+start_index][j] = single_pixel_convolution(input_image, i, j, kernel, kernel_height, kernel_width)
            if (output_matrix[i+start_index][j]>255):
                output_matrix[i+start_index][j] = 255
            elif (output_matrix[i+start_index][j]<0):
                output_matrix[i+start_index][j] = 0
    
    for i in range (wrap_index, reduced_height):
        for j in range (0,reduced_width):
            output_matrix[i-wrap_index][j] = single_pixel_convolution(input_image, i, j, kernel, kernel_height, kernel_width)
            if (output_matrix[i-wrap_index][j]>255):
                output_matrix[i-wrap_index][j] = 255
            elif (output_matrix[i-wrap_index][j]<0):
                output_matrix[i-wrap_index][j] = 0

input_image_matrix = np.asarray(Image.open("source.jpg").convert("L"))

input_image_height = len(input_image_matrix)
input_image_width = len(input_image_matrix[0])
#defining kernel for convolution
convolution_kernel_width = 3
convolution_kernel_height = 3
convolution_kernel = np.array([[1,0,0], [0,0,0], [0,0,0]])

#kernel for cropping images
# convolution_kernel_width = 30
# convolution_kernel_height = 30
# convolution_kernel = np.zeros(shape=(convolution_kernel_width,convolution_kernel_height))
# convolution_kernel[0][0]=1
convolution2D_for_wrap_effect(input_image_matrix, input_image_width, input_image_height, convolution_kernel, convolution_kernel_height, convolution_kernel_width)
#convolution_2D(input_image_matrix, input_image_width, input_image_height, convolution_kernel, convolution_kernel_height, convolution_kernel_width)
print (input_image_matrix)
print (output_matrix)
#convert numpy matrix back to image
output_image = Image.fromarray(output_matrix)
output_image.show()
output_image.save('target_4.jpeg')
