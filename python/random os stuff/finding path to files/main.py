import os

ppath = r"Z:\\Documents\\Ved\\github scratchpad"
path = r"Z:\Documents\Ved\github scratchpad\very very old projects\Password Generator That Stores Passwords\data.csv"

relative_path = os.path.relpath(path, ppath)
print(relative_path)
