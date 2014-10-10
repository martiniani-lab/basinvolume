# Moves output of BV computation from cluster to a defined remote location.
#
# Intended procedure:
# 1. Check: Computation terminated properly.
# 2. Check: Destination can be rached / is a valid path / dir exists.
# 3. Check: Enough disc space available at destination.
# 4. Do: At the end of PT, or manually, scp the data in batch mode, roughly with
#    scp -BCvr explore_bv_jammed_packing500 "bazinga:/media/My\ Passport/LinuxPartition/n32_phi50_phi88_3D"
# 5. Check that the transfer was successful, i.e. that data at origin and destination is identical.
# 6. Do: Erase folder at origin.
#
# Input parameters: destination location
# Output: data moving success?
# 