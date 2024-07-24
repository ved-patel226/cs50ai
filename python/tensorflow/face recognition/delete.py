import os, shutil
from termcolor import cprint

folder = "images"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
            cprint("Deleted File: " + file_path, "green", attrs=["bold"])
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            cprint("Deleted directory: " + file_path, "green", attrs=["bold"])
    except Exception as e:
        cprint(
            "Failed to delete %s. Reason: %s" % (file_path, e), "green", attrs=["bold"]
        )
