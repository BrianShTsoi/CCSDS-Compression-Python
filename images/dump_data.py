# Yong Da Li
# Saturday, May 14, 2022
# dumping .mat file to text files


# file format
# xdim, ydim, zdim
# 1,2,3,4 <-- each row is a row of data
# ..
# ..
# .. more rows until it makes up a column

# -- the "--" separate denotes a new frame

import scipy.io
import numpy as np

in_file = "Indian_pines.mat"
out_file = "data_145_145_220.txt"

# load data
matlab_data = scipy.io.loadmat(in_file)
print(matlab_data.keys())
data = matlab_data['indian_pines']

# write header 
[xlen, ylen, zlen] = data.shape
f = open(out_file, "w") # overwrite existing data
f.write("{}, {}, {}\n".format(xlen, ylen, zlen))
f.close()

# open in append mode append
f = open(out_file, "a")

# data size is 145, 145, 220
# slice into 220x of 145x145 images
for i in range (0, len(data[:,:,0])):
	# get frame
	z_slice = data[:,:,i]

	# format and write each frame to file as a single line
	for j in range (0, len(z_slice)):
		row = z_slice[j].tolist()
		row = str(row)
		row = row.replace('[', '').replace(']', '') # remove array brackets
		f.write(row) # expects string format
		f.write("\n")

	# f.write("--\n") # write nothing between frames

f.close()