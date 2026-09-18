import gsd.hoomd
import os
import glob

# use find_params to get params
sim_pattern = 'Shear-*-DPD.gsd'
sim_list = glob.glob(sim_pattern)
if not sim_list:
  print("ERROR: no GSD files found to save last frame of")
  exit()
else:
  sim_list.sort(key=os.path.getctime) # sort list by file creation time
  input_filename = sim_list[-1]

# convert input_filename into output_filename
sim_style = os.path.basename(input_filename)
sim_style = os.path.splitext(sim_style)[0]
if sim_style != "":
  output_filename = sim_style+'-lastframe.gsd'
else:
  output_filename = 'lastframe.gsd'


# Read the GSD file
with gsd.hoomd.open(input_filename, 'r') as gsd_file:
    # Get the last frame
    last_frame = gsd_file[-1]

    # Save the last frame as a new GSD file
    with gsd.hoomd.open(output_filename, 'w') as output_file:
        output_file.append(last_frame)

print(f"Last frame saved to {output_filename}")

