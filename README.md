# self-organizing-feature-maps-clustering
#### An unsupervised clustering algorithm, self-organizing feature maps is used for two applications, organizing colors by clustering them and compressing an image by clustering the image frames during training on the sofm network, then saving it into a binary file and using the compressed binary file to get back the original image.

Clustering is a task of grouping set of objects with similar features or data points and separate them from other groups. Clustering is an important unsupervised learning technique with many use cases in pattern recognition, image analysis and machine learning. Using self-organizing feature map (SOFM) also known as Kohonen network, to cluster provides aspect of topologies which helps preserve relative positioning by using a neighborhood function. SOFMs are also used for dimensionality reduction as it produces a low dimensional output space from multi-dimension input space. SOFM’s apply competitive learning to find patterns by finding the best matching unit (BMU) to the input vector in the network. Weights are randomly generated at the beginning, the best matching unit in the network for a given pattern or input is found during training where the closest unit to input is chosen and its weights are updated according to learning rate. This BMU is used to update the neighbouring units within the neighbourhood radius provided. In an iteration all units are updated according to the neighborhood multiplier and learning rate. If the unit is BMU then the neighborhood multiplier is 1 giving it full update. The BMU nodes will be updated more than the neighbouring nodes as neighbourhood multiplier dampens the affect of weight update as moving away from centroid. After certain number of iterations, network which is a collection of vectors mapping from input acts as exemplars reflecting the input and resembling the pattern.

### Colour Organization

Kohonen networks on random colours. A N `×` N grid of vectors, initially randomized is trained on random colour vectors. Each individual pixel visualizes a map as an image/panel or an individual node in the network which are then clustered according to their colour. 

<img src="https://github.com/SharmaSapan/self-organizing-feature-maps-clustering/blob/master/colour1/colour_gif.gif" width="300" height="300" /> <img src="https://github.com/SharmaSapan/self-organizing-feature-maps-clustering/blob/master/colour2/colour_gif.gif" width="300" height="300" /> <img src="https://github.com/SharmaSapan/self-organizing-feature-maps-clustering/blob/master/colour3/colour_gif.gif" width="300" height="300" />

To run part 1 run main file and enter 1 and follow command line prompts for parameters.

Three tests are conducted with three different training parameters and results stored in colour and colour2.
Interval of image save can be mentioned both have 500.
colour1 parameters: N `×` N- 600 `×` 600, Epochs- 3000, Neighbour radius- 20, Learning rate- 1.
colour2 parameters: N `×` N- 800 `×` 800, Epochs- 3000, Neighbour radius- 40, Learning rate- 0.7.
colour3 parameters: N `×` N- 1000 `×` 1000, Epochs- 3000, Neighbour radius- 60, Learning rate- 0.9.

Sample input:
1
600
600
3000
30
1.0
colour3
200

A GIF is stored showing visualizations along with images of your interval.

### Image Compression 

Image compression is done using Self-organizing feature maps by clustering the frames and using network indices mapped to image frames indices and stored as a binary codebook. Codebook has all the necessary information to regenerate the image and test the compression.

Image is converted to grayscale and is divided into frames. A unit class is used to initialize SOFM nodes in network and store its variables. Compression algorithm then initializes a Best matching unit and finds its Euclidean distance from the input vector. Node with the least distance in the network to input is then chosen. Weight update is performed on the node by factor of learning rate as in this code: new_weight = BMU_frame.sofm[b_index] + lr*(input_frame[b_index]-BMU_frame.sofm[b_index]). No neighborhood multiplier is used since that will be one due to distance from BMU to BMU itself. Then a for loop is used to iterate through the four quadrants around BMU node according to the given radius. On Y-axis, weight update starts at BMU y-coordinate – neigh_radius till BMU y-coordinate + neigh_radius. And moving to X-axis both on left side and right side, weight update is done within the perimeter of the circle around BMU.  Neighbourhood multiplier is used to update weight which is found by code: neigh_multi = pow(math.exp(1),(-1*(((sofm_map[xl][y].getEDistance(BMU_frame.sofm))*(sofm_map[xl][y].getEDistance(BMU_frame.sofm)))/(2*neigh_size*neigh_size)))).

After training, frames are matched to the network to find index of the best matching node to each frame using Euclidean distance. And if the SOFM map is 4*4 indices are converted to store in one nybble per frame fashion. Then the codebook is stored in binary file in a format:  Width of image(in frames), height of image(in frames), width of single frame, height of single frame, width of network, height of network, map result from SOFM units in 1D, and indices of map which matches best to each frame of the image. Binary file is then read to reconstruct the image from the generated codebook. The high nibble and low nibble are separated if the network is 4*4 and mapped to SOFM to reconstruct image. Accuracy is calculated using the square root of sum of difference squared between pixel intensities of Reconstructed image and original grayscale image. Binary files, images screenshot and command line output are stored in compressed_data folder with the file names according to their parameters. A window with the reconstructed image pops up at the end of run.

Tests were conducted to compare different parameters and image complexities.

Code was tested on four different images with their own unique sets of features. P1 image is of a furry fox in snow. It was compressed using 8 frame size (which make 8*8 frames) on a 16*16 config, with 0.9 learning rate with 10,000 epochs for different neighbour radius. A radius of 70 generated higher root squared absolute sum of distance 35800 to 46982 by about 31% to radius with 150. It was also evident from the image as screenshot shown side by side of Neighbouring distance of 70, then 150.

<img src="https://github.com/SharmaSapan/self-organizing-feature-maps-clustering/blob/master/1git.png" width="700" height="300" /> 

Image was further distorted when the frame size of 16*16 was used. Higher neighbourhood function allows more exploration before limiting weight update using neighborhood multiplier by getting smaller over iterations and converging to just the winning neuron at the end. Both image generated and binary files were smaller in size compared to original and grayscale photos. For 4*4 there was a huge difference in file size compared to original. Also, file size for 8*8 frames was observed to be smaller than 16*16.

In Conclusion, I can say that parameters play important role in self-organizing maps and performance is critically dependent on them. Parameters should be selected experimentally with which accuracy improves significantly. Self-organizing maps can be great tool to map high dimensions to low dimensions and reconstruct as it was shown in this assignment example.

To run part 2 run main file and enter 2 and follow command line prompts for parameters.
To test your image, place it in the main folder of this assignment i.e. assign_2
A window with reconstructed image from binary file pops up at the end of run. Tests specification is in report. I was unable to save image, but i have collected screenshots.

Sample input:
2
8
16
p1.jpg
10000
150
0.9
filename

Output accuracy is printed in commandline.
