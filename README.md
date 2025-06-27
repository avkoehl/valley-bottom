# Valley-Bottom

Code for delineating valley bottoms in a digital elevation model (DEM) using Python.

Valley bottoms are defined as the parts of valley floors that are within a set elevation
above the stream. Elevation above the stream is a user defined parameter that
can be scaled based on stream attributes such as stream order and gradient. 

## Installation

### Prerequisites

- Python 3.10 or higher
- [Poetry (package manager)](https://python-poetry.org/)

### Installing from Github

1. Clone the repository
```bash
git clone git@github.com:avkoehl/valley-bottom.git
cd valley-bottom
```

2. Install dependencies using Poetry
```bash
poetry install
```
## Usage

### Basic Usage
Import and use the modules in your python code:

```python
from valley_bottom import extract_valley_bottom
from valley_bottom import Config
from valley_bottom import load_sample_dem
from valley_bottom import load_sample_flowlines


config = Config()

dem = load_sample_dem()
flowlines = load_sample_flowlines()
valley_bottom = extract_valley_bottom(dem, flowlines, config, return_basins=False)
```

### Configuration Options

To customize parameters, create a `Config` object and modify the
parameters:

```python
config = Config()
config.hand_hg = 3 # Set the HAND threshold for high gradient streams to 3
```

For more information on the parameters:
```python
help(config)
```


## Contact

Arthur Koehl  
avkoehl at ucdavis .edu
