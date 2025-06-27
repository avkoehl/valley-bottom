import matplotlib.pyplot as plt

from valley_bottom import extract_valley_bottom
from valley_bottom import Config
from valley_bottom import load_sample_dem
from valley_bottom import load_sample_flowlines

config = Config()

dem = load_sample_dem()
flowlines = load_sample_flowlines()

bottom = extract_valley_bottom(dem, flowlines, config, return_basins=False)

bottom.plot(figsize=(10, 10))
plt.show()
