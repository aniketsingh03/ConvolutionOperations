# Convolutional Operations
This is just a test of operations with the convolution algorithm.

## Jist to the algorithm
A convolution matrix filter uses two matrices, one of them is a matrix of the input image, whereas the other one is a matrix we choose based on the effect we want, and this matrix is called as the **kernel**.<br>

We perform a mathematical convolution between these matrices which is basically the multiplication of both these matrices pixelwise, while sliding the kernel over the entire image matrix. Sliding the kernel over different regions may result in different interesting patterns. Famous graphics editors like **gimp** work on this algorithm.<br>

## Results
<center><b>ORIGINAL IMAGE</b></center><br>
<center><img src="source.jpg" width="400px"></center>

- This is a more sharper version of the original image. The image shown below was generated by sweeping the kernel over the entire image matrix and by taking the kernel matrix as `[[-2,-1,0],[-1,1,1],[0,1,2]]`.<br>

<center><img src="target_1.jpeg" width="400px"></center><br>

- The image shown below was generated by sweeping the kernel over the entire image matrix and by taking the kernel matrix as `[[0,1,0],[1,-4,1],[0,1,0]]`.<br>

<center><img src="target_2.jpeg" width="400px"></center><br>

- **Cropped Result** - On choosing a kernel of larger size, for example of size `30x30` for an input image of size `200x200`, we may see the cropping effect significantly.<br>

<img src="source_2.jpg"><img src="target_3.jpeg"><br>

- **Image Wrap** - Taking convolution at different segments can produce wrapping effect.<br>

<img src="source.jpg" width="400px"><img src="target_4.jpeg" width="400px">