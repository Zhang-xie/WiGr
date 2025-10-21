# CSI Dataset

A comprehensive Channel State Information (CSI) dataset containing raw CSI data, processed amplitude and phase data, along with corresponding labels for activity, environment, location, and user identification.

## Dataset Structure

```
dataset/
├── csi_data_raw/          # Raw CSI data (1422 files)
├── csi_data_amp/          # CSI amplitude data (1422 files)  
├── csi_data_pha/          # CSI phase data (1422 files)
├── csi_label_act/         # Activity labels
├── csi_label_env/         # Environment labels
├── csi_label_loc/         # Location labels
├── csi_label_user/        # User labels
└── parsing.py             # Data loading utilities
```

## Data Format

All data is stored in **Zarr format** for efficient storage and access:

- **Data Arrays**: Raw, amplitude, and phase CSI data stored as compressed Zarr arrays
- **Labels**: Categorical labels stored as 1D Zarr arrays with shape `(2844,)`
- **Compression**: LZ4 compression with blosc compressor for optimal performance

## Data Types

### CSI Data
- **Raw CSI Data** (`csi_data_raw/`): Original channel state information
- **Amplitude Data** (`csi_data_amp/`): CSI amplitude values
- **Phase Data** (`csi_data_pha/`): CSI phase values

### Labels
- **Activity Labels** (`csi_label_act/`): Human activity classifications
- **Environment Labels** (`csi_label_env/`): Environmental conditions
- **Location Labels** (`csi_label_loc/`): Spatial location information
- **User Labels** (`csi_label_user/`): User identification data

## Usage

### Prerequisites

```bash
pip install zarr numpy
```

### Loading Data

Use the provided `CSIDataLoader` class in `parsing.py`:

```python
from parsing import CSIDataLoader

# Initialize loader
loader = CSIDataLoader()

# Load CSI data
raw_data = loader.load_csi_data_raw()
amp_data = loader.load_csi_data_amp()
pha_data = loader.load_csi_data_pha()

# Load labels
activity_labels = loader.load_csi_label_act()
env_labels = loader.load_csi_label_env()
location_labels = loader.load_csi_label_loc()
user_labels = loader.load_csi_label_user()
```

### Direct Zarr Access

For more control, access data directly:

```python
import zarr
import numpy as np

# Load any dataset
zarr_array = zarr.open('csi_data_raw', mode='r')
data = np.array(zarr_array)

# Access metadata
print(f"Shape: {zarr_array.shape}")
print(f"Data type: {zarr_array.dtype}")
print(f"Chunks: {zarr_array.chunks}")
```

## Dataset Statistics

- **Total Samples**: 2,844
- **Data Files**: 1,422 files per data type
- **Label Shape**: (2844,) - one label per sample
- **Data Type**: 64-bit integers for labels
- **Compression**: LZ4 with blosc compressor


## File Information

- **Format**: Zarr v2
- **Compression**: LZ4 (clevel=5, shuffle=1)
- **Chunk Size**: 2844 samples per chunk
- **Fill Value**: 0

## Notes

- All data is stored in compressed format for efficient storage
- Labels are aligned with corresponding CSI data samples
- The dataset contains 1,422 data collection sessions
- Each session contains multiple CSI measurements
