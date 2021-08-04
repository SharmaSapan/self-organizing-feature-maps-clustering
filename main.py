import math
import random
import os
# pillow image handling library
from PIL import Image
import imageio  # to create gif in part 1
import numpy as np  # to show image in part 2

# Pixel class to store pixel object and its properties.
class Pixel:
    x = 0  # x-coordinate of pixel
    y = 0  # y-coordinate of pixel

    # create pixel objects using rgb, if no values given then random colors are assigned
    def __init__(self, *args):
        # random pixel generation if no value during object initialization
        if len(args) == 0:
            self.r = random.random()
            self.g = random.random()
            self.b = random.random()
        # rgb values given during call
        if len(args) > 0:
            self.r = args[0]
            self.g = args[1]
            self.b = args[2]

    # to update rgb values of the object
    def setNewRGB(self, new_r, new_g, new_b):
        self.r = new_r
        self.g = new_g
        self.b = new_b

    # find the euclidean distance between this pixel and the given pixel
    def getEDistance(self, p):
        distance = math.sqrt(((self.r - p.r)**2) + ((self.g - p.g)**2) + ((self.b - p.b)**2))
        return distance


# SOFM for colour organization
def color(map_width, map_height, arr, epochs, neigh_radius, learning_rate, folder, interval):
    print("Color Organization started....")
    for i in range(epochs):
        lr = learning_rate*pow(math.exp(1), (-1*(i/epochs)))
        neigh_size = math.ceil(neigh_radius*pow(math.exp(1), (-1*(i/epochs))))
        input_pixel = Pixel() # random input space pixel which will be generated after each epoch
        # pixel to find best matching unit initially r,g,b = 0 and x,y =0
        BMU_pixel = Pixel(0, 0, 0)
        BMU_distance = float('inf') # to calculate best distance between input pixel and BMU
        #print(BMU_pixel.r,BMU_pixel.x,BMU_pixel.y)
        for j in range(map_width):
            for k in range(map_height):
                if(BMU_distance > input_pixel.getEDistance(arr[j][k])): # if pixel distance from BMU is less, store pixel as the BMU
                    BMU_distance = input_pixel.getEDistance(arr[j][k])
                    BMU_pixel = arr[j][k]
                    BMU_pixel.x = j
                    BMU_pixel.y = k

        # colour update at arr with best matching colour in input space by a factor of learning rate
        b_r = BMU_pixel.r + lr * (input_pixel.r-BMU_pixel.r) # neigbourhood mulitplyer not needed as we are updating at BMU
        b_g = BMU_pixel.g + lr * (input_pixel.g-BMU_pixel.g)
        b_b = BMU_pixel.b + lr * (input_pixel.b-BMU_pixel.b)
        #print(arr[BMU_pixel.x][BMU_pixel.y].r)
        arr[BMU_pixel.x][BMU_pixel.y].setNewRGB(b_r,b_g,b_b)
        #print(arr[BMU_pixel.x][BMU_pixel.y].r, BMU_pixel.r,BMU_pixel.x,BMU_pixel.y)
        #print(y)

        # from lower quadrant to upward quadrant in circle to update height(y-axis)
        for y in range(BMU_pixel.y-neigh_radius,BMU_pixel.y+neigh_radius):
            # updating pixel values at width(x-axis) to the left from BMU according to radius
            xl = BMU_pixel.x - 1
            # get values left of current y-axis till neighbour radius
            while (((xl-BMU_pixel.x)*(xl-BMU_pixel.x) + (y-BMU_pixel.y)*(y-BMU_pixel.y)) <= (neigh_radius*neigh_radius)):
                if(xl<map_width and xl>=0 and y<map_height and y>=0):
                    neigh_multi = pow(math.exp(1),(-1*(((arr[xl][y].getEDistance(BMU_pixel))*(arr[xl][y].getEDistance(BMU_pixel)))/(2*neigh_size*neigh_size))))
                    rl = arr[xl][y].r + neigh_multi * lr * (input_pixel.r-arr[xl][y].r) # if BMU then the difference will be 0 making no change to the BMU pixel colour
                    gl = arr[xl][y].g + neigh_multi * lr * (input_pixel.g-arr[xl][y].g)
                    bl = arr[xl][y].b + neigh_multi * lr * (input_pixel.b-arr[xl][y].b)
                    arr[xl][y].setNewRGB(rl,gl,bl) # update colour of neighbours by a factor neighbourhood multiplyer and learning rate
                xl = xl-1
            # updating pixel values at width to the right from BMU according to radius
            xr = BMU_pixel.x
            # get values right of current y-axis till neighbour radius
            while (((xr-BMU_pixel.x)*(xr-BMU_pixel.x) + (y-BMU_pixel.y)*(y-BMU_pixel.y)) <= (neigh_radius*neigh_radius)):
                if(xr<map_width and xr>=0 and y<map_height and y>=0):
                    neigh_multi = pow(math.exp(1),(-1*(((arr[xr][y].getEDistance(BMU_pixel))*(arr[xr][y].getEDistance(BMU_pixel)))/(2*neigh_size*neigh_size))))
                    rr = arr[xr][y].r + neigh_multi * lr * (input_pixel.r-arr[xr][y].r)
                    gr = arr[xr][y].g + neigh_multi * lr * (input_pixel.g-arr[xr][y].g)
                    br = arr[xr][y].b + neigh_multi * lr * (input_pixel.b-arr[xr][y].b)
                    arr[xr][y].setNewRGB(rr,gr,br)
                    #print(x,y,arr[x][y].r)
                xr = xr+1
        # to save image after user provided intervals
        if i % interval == 0:
            initial = Image.new('RGB', (map_width, map_height), "black")
            pix = initial.load()
            for row in range(map_width):
                for col in range(map_height):
                    pix[row,col] = (int(arr[row][col].r*255),int(arr[row][col].g*255),int(arr[row][col].b*255))
            saveloc = str(i+1) + "th_epoch.jpg"
            initial.save(os.path.join(folder, saveloc))
            print(i, "th epoch saved")
    initial.save(os.path.join(folder, "final.jpg"))


class Unit:
    x = 0  # x-coordinate of unit
    y = 0  # y-coordinate of unit

    # create single sofm unit objects randomly or using 0's, if no values given then random values are assigned
    def __init__(self, frame_size):
        # random weight generation during object initialization
        self.sofm = random.sample(range(0, 256), frame_size) # 64 or 256 depending on frame

    # to update sofm values of the object
    def setNew(self, index, value):
        self.sofm[index] = value

    # find the euclidean distance between this unit and the given frame
    def getEDistance(self, frame):
        distance = 0
        for index in range(len(frame)):
            distance += ((self.sofm[index] - frame[index])**2)
        sq = math.sqrt(distance)
        return sq


def compression(epochs, learning_rate, neigh_radius, frames, frame_size, sofm_config, sofm_map):
    for i in range(epochs):
        lr = learning_rate*pow(math.exp(1), (-1*(i/epochs)))
        neigh_size = math.ceil(neigh_radius*pow(math.exp(1), (-1*(i/epochs))))
        input_frame = random.choice(frames)
        BMU_frame = Unit(frame_size**2) # random at first but will have copy from network during BMU scan
        BMU_distance = float('inf')
        for map_row in range(sofm_config):
            for map_col in range(sofm_config):
                if(BMU_distance > (sofm_map[map_row][map_col]).getEDistance(input_frame)):
                    BMU_distance = sofm_map[map_row][map_col].getEDistance(input_frame)
                    BMU_frame = sofm_map[map_row][map_col]
                    BMU_frame.x = map_row
                    BMU_frame.y = map_col

        for b_index in range(len(BMU_frame.sofm)):
            new_weight = BMU_frame.sofm[b_index] + lr*(input_frame[b_index]-BMU_frame.sofm[b_index])
            sofm_map[BMU_frame.x][BMU_frame.y].setNew(b_index, new_weight)

        for y in range(BMU_frame.y-neigh_radius,BMU_frame.y+neigh_radius):
            xl = BMU_frame.x - 1
            # get values left of current y-axis till neighbour radius
            while (((xl-BMU_frame.x)*(xl-BMU_frame.x) + (y-BMU_frame.y)*(y-BMU_frame.y)) <= (neigh_radius*neigh_radius)):
                if(xl<sofm_config and xl>=0 and y<sofm_config and y>=0):
                    neigh_multi = pow(math.exp(1),(-1*(((sofm_map[xl][y].getEDistance(BMU_frame.sofm))*(sofm_map[xl][y].getEDistance(BMU_frame.sofm)))/(2*neigh_size*neigh_size))))
                    for xl_index in range(len(BMU_frame.sofm)):
                        new_n_weight = sofm_map[xl][y].sofm[xl_index] + neigh_multi*lr*(input_frame[xl_index]-sofm_map[xl][y].sofm[xl_index])
                        sofm_map[xl][y].setNew(xl_index, new_n_weight)
                xl = xl-1

            xr = BMU_frame.x
            # get values right of current y-axis till neighbour radius arc
            while (((xr-BMU_frame.x)*(xr-BMU_frame.x) + (y-BMU_frame.y)*(y-BMU_frame.y)) <= (neigh_radius*neigh_radius)):
                if(xr<sofm_config and xr>=0 and y<sofm_config and y>=0):
                    neigh_multi = pow(math.exp(1),(-1*(((sofm_map[xr][y].getEDistance(BMU_frame.sofm))*(sofm_map[xr][y].getEDistance(BMU_frame.sofm)))/(2*neigh_size*neigh_size))))
                    for xr_index in range(len(BMU_frame.sofm)):
                        new_n_weight = sofm_map[xr][y].sofm[xr_index] + neigh_multi*lr*(input_frame[xr_index]-sofm_map[xr][y].sofm[xr_index])
                        sofm_map[xr][y].setNew(xr_index, new_n_weight)
                xr = xr+1


def main():
    part = int(input("Enter 1 for part-1 Colour, Enter 2 for part-2 Compression: "))
    if part == 1:
        map_width = int(input("Enter N width for grid (ideally 200-800): "))
        map_height = int(input("Enter N height for grid: "))
        arr = []
        initial = Image.new('RGB', (map_width, map_height), "black") # to create initial image
        pix = initial.load()
        # initialize a 2-D list of Pixel objects(should have used a better way, I know)
        for width in range(map_width):
            col = []
            for height in range(map_height):
                col.append(Pixel())
                col[height].x = width
                col[height].y = height
            arr.append(col)

        # to access values use arr[i][j].x (x for x-coordinate, y for y-coordinate, r,g,b for r,g,b random color respectively)
        epochs = int(input("Enter number of epochs: "))
        neigh_radius = int(input("Enter integer radius: "))
        learning_rate = float(input("Enter learning rate: "))
        folder = input("Enter folder name to store image: ")
        interval = int(input("Enter integer interval for image save like 50 or 100 epochs: "))
        os.mkdir(folder) # create folder with user provide name
        for width_p in range(map_width): # paint image
            for height_p in range(map_height):
                pix[width_p,height_p] = (int(arr[width_p][height_p].r*255),int(arr[width_p][height_p].g*255),int(arr[width_p][height_p].b*255))
        initial.save(os.path.join(folder,"1initial_input_space.jpg"))
        color(map_width, map_height, arr, epochs, neigh_radius, learning_rate, folder, interval) # sofm engine
        images = [] # store gif
        gifloc = os.path.join(folder, 'colour_gif.gif')
        for filename in os.listdir(folder):
            images.append(imageio.imread(os.path.join(folder, filename)))
            imageio.mimsave(gifloc, images)
        print("End of run")

    if part == 2:
        frame_size = int(input("Enter frame size of 8 or 16: "))
        sofm_config = int(input("Enter SOFM map config '16' for 16*16 or '4' for 4*4: "))
        image_name = input("Enter name of image file with extension to run test on: ")
        epochs = int(input("Enter number of epochs: "))
        neigh_radius = int(input("Enter neighbour radius for SOFM: "))
        learning_rate = float(input("Enter learning rate: "))
        savfile = input("Enter name of the file to save binary data: ") + ".bin"
        print("Running...")
        im = Image.open(image_name).convert('L')  # load image
        im.save("gray-"+image_name)
        imgwidth, imgheight = im.size
        v = list(im.getdata())
        frames = []
        v_index = 0
        for frame_index in range(int(len(v)/(frame_size**2))): # divide data by frame size**2
            frame = []
            for item_index in range(frame_size**2): # store per frame
                frame.append(v[item_index+v_index])
            v_index += (frame_size**2)
            frames.append(frame)
        sofm_map = []  # create a sofm map of sofm_config*sofm_config
        for sofm_index in range(sofm_config):
            sofm_col = []
            for col_index in range(sofm_config):
                sofm_col.append(Unit(frame_size**2))  # initialize vector units of sofm with random vectors of frame_size**2
            sofm_map.append(sofm_col)
        # training on the image
        compression(epochs, learning_rate, neigh_radius, frames, frame_size, sofm_config, sofm_map)
        trained_sofm_1D = []  # convert sofm map from 2D to 1D to map frames from image for reference in bin
        for net_index in range(len(sofm_map)):
            for net_col in range(len(sofm_map[0])):
                trained_sofm_1D.append(sofm_map[net_index][net_col])
        # reference frames which are best matching index of sofm unit to images from frame
        frame_reference = []
        for img_frame_index in range(len(frames)):
            BMU_dist = float('inf')
            best_index = 0
            d = 0  # distance between frame and unit from sofm
            for unit_index in range(len(trained_sofm_1D)):
                d = trained_sofm_1D[unit_index].getEDistance(frames[img_frame_index])
                if(BMU_dist > d):
                    BMU_dist = d
                    best_index = unit_index
            frame_reference.append(best_index)

        if sofm_config == 4:
            # conversion to store two indices in one byte for sofm configuration of 4*4
            frame_reference_nibble_list = []
            for inde in range(0,len(frame_reference),2): # get 0 and 1, then 2 and 3rd index
                frame_reference_nibble = frame_reference[inde] # for if last and odd length store as it is
                if inde!=len(frame_reference)-1:
                    frame_reference_nibble = (frame_reference[inde] << 4) | (frame_reference[inde+1])
                frame_reference_nibble_list.append(frame_reference_nibble)

    # NOTE: Code can handle any format but we can only store 0-255 in binary file,
        # to store image resolution/frame size binary dump only works on max 2040*2040 picture. Please test picture in that resolution.

        # Codebook format: Width of image(in frames), height of image(in frames), width of single frame,
        # height of single frame, width of network, height of network, map result from sofm units in 1D,
        # and indices of map which matches best to each frame of the image
        width_image_in_frames_to_byte = [int(imgwidth/frame_size)] # binary file can only handle 2040*2040
        height_image_in_frames_to_byte = [int(imgheight/frame_size)]
        width_frame_to_byte = [frame_size]
        height_frame_to_byte = [frame_size]
        width_network_to_byte = [sofm_config]
        height_network_to_byte = [sofm_config]
        map_to_byte = []  # appending individual values of sofm unit in 1D
        for sofm_codebook in trained_sofm_1D:
            for val in sofm_codebook.sofm:
                map_to_byte.append(int(val))

        # data is stored on binary file name provided by user during runtime.
        with open(savfile,"wb") as binary_file:
            binary_file.write(bytes(width_image_in_frames_to_byte))
            binary_file.write(bytes(height_image_in_frames_to_byte))
            binary_file.write(bytes(width_frame_to_byte))
            binary_file.write(bytes(height_frame_to_byte))
            binary_file.write(bytes(width_network_to_byte))
            binary_file.write(bytes(height_network_to_byte))
            binary_file.write(bytes(map_to_byte))  # saving map to bytes file
            if sofm_config == 4:  # store one nibble per byte
                binary_file.write(bytes(frame_reference_nibble_list))  # saving codebook indices, 2 indices per byte
            else: # else if sofm_config 16 then store one per byte
                binary_file.write(bytes(frame_reference))  # saving codebook indices

        # reading data after storing
        with open(savfile,"rb") as binary_file:
            data = binary_file.read()

        print("Codebook saved")

        # converting binary to usable format.
        convert_width_image_in_frames_to_byte = convert_height_image_in_frames_to_byte= convert_width_frame_to_byte= convert_height_frame_to_byte= convert_width_network_to_byte= convert_height_network_to_byte = 0
        convert_sofm_map_values = []
        convert_frame_reference = []
        for i in range(len(data)):
            if i == 0:
                convert_width_image_in_frames_to_byte = data[i]
            if i == 1:
                convert_height_image_in_frames_to_byte = data[i]
            if i == 2:
                convert_width_frame_to_byte = data[i]
            if i == 3:
                convert_height_frame_to_byte = data[i]
            if i == 4:
                convert_width_network_to_byte = data[i]
            if i == 5:
                convert_height_network_to_byte = data[i]
            # getting sofm map values
            if i > 5 and i <= ((convert_width_network_to_byte*convert_height_network_to_byte*convert_width_frame_to_byte*convert_height_frame_to_byte)+5):
                convert_sofm_map_values.append(data[i])
            # getting codebook indices
            if i > ((convert_width_network_to_byte *convert_height_network_to_byte*convert_width_frame_to_byte*convert_height_frame_to_byte)+5):
                if sofm_config == 4:
                    convert_frame_reference.append((data[i] >> 4)) # high nibble
                    convert_frame_reference.append((data[i] & 0x0F)) # low nibble
                else:
                    convert_frame_reference.append(data[i]) # full byte if sofm configuration is 16*16

        image_width = convert_width_image_in_frames_to_byte*convert_width_frame_to_byte
        image_height = convert_height_image_in_frames_to_byte*convert_height_frame_to_byte
        network_units = convert_width_frame_to_byte*convert_height_frame_to_byte #64 or 256
        convert_map_to_2D = []  # to store sofm map back into 2D
        unit_vector_i = 0
        for map_index in range(convert_width_network_to_byte*convert_height_network_to_byte):
            convert_unit = []
            for iterate in range(network_units):
                convert_unit.append(convert_sofm_map_values[iterate+unit_vector_i])
            unit_vector_i += network_units
            convert_map_to_2D.append(convert_unit)

        # reconstruct image frames using sofm map and codebook indices
        reconstructed_image_frames = [] # which is similar to frames list made from image data
        for convert_index in convert_frame_reference:
            reconstructed_image_frames.append(convert_map_to_2D[convert_index])  # add frames from map using index

        # to see difference between original and compressed
        accuracy_sum = 0
        for frame_in in range(len(reconstructed_image_frames)):
            for pixel_index in range(len(reconstructed_image_frames[0])):
                accuracy_sum += ((reconstructed_image_frames[frame_in][pixel_index]-frames[frame_in][pixel_index])**2)
        accuracy = math.sqrt(accuracy_sum)
        print(accuracy)

        # to show image
        hh = np.asarray(reconstructed_image_frames).reshape(image_height,image_width)
        imii = Image.fromarray(hh)
        imii.show()
        # imii.save("savetest.png") stopped working

if __name__ == '__main__' :
    main()