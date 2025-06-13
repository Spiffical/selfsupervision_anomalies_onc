import numpy as np

# Find consecutive points in the "data" variable and produce a
#  two-column output with start and end indices of these consecutive points. 

def segment2(data):

    # Assuming 'data' is a list or NumPy array containing the indices

    # Convert 'data' to a NumPy array
    data = np.array(data)

    # Find differences between consecutive elements
    seg = np.diff(data)

    # Add a 2 at the end to ensure the last segment is included
    seg = np.append(seg, 2)

    # Find indices where the segment changes
    seg = np.where((seg > 1) | (seg < -1))[0]

    # Find the number of elements in each segment
    seg_num = np.diff(seg)

    # Get the starting indices of each segment
    seg_loc = seg[:-1] + 1

    # Manually add the starting index of the first segment
    seg_loc = np.insert(seg_loc, 0, 0)

    # Get the data for the starting and ending indices of each segment
    data1 = data[seg_loc]
    data2 = np.append(data[seg_loc[1:] + seg_num - 1], data[-1])
    #data2 = data[seg_loc + seg_num - 1]

    # Combine 'data1' and 'data2' to create the final output
    dataout = np.column_stack((data1, data2))

    # The 'dataout' variable now contains the start and end indices of consecutive points
    return dataout