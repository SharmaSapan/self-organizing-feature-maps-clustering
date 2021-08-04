Part - 1:

To run part 1 run main file and enter 1 and follow command line prompts for parameters.

Three tests are conducted with three different training parameters and results stored in colour and colour2.
Interval of image save can be mentioned both have 500.
colour1 parameters: N*N- 600*600, Epochs- 3000, Neighbour radius- 20, Learning rate- 1.
colour2 parameters: N*N- 800*800, Epochs- 3000, Neighbour radius- 40, Learning rate- 0.7.
colour3 parameters: N*N- 1000*1000, Epochs- 3000, Neighbour radius- 60, Learning rate- 0.9.

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
(Note: I was trying to build a tkinter window to show visualizations but was not working properly so I resorted to a GIF stored at the end of run.)

Part - 2:

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