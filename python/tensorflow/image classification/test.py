import time
from alive_progress import alive_it

for item in alive_it(range(100), spinner="waves"):
    time.sleep(0.01)
