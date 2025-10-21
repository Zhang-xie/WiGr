import zarr
import numpy as np


class CSIDataLoader:
    """A class to load CSI data from Zarr format files"""
    
    def __init__(self):
        """Initialize the CSI data loader"""
        pass
    
    def load_csi_label_env(self):
        """Load the CSI environment label data from Zarr format"""
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_label_env', mode='r')
            
            print("Zarr Array Info:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            print(f"Compressor: {zarr_array.compressors}")
            
            # Convert to numpy array
            data = np.array(zarr_array)
            
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"First 10 values: {data[:10]}")
            print(f"Last 10 values: {data[-10:]}")
            print(f"Unique values: {np.unique(data)}")
            print(f"Value range: {data.min()} to {data.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def load_csi_label_user(self):
        """Load the CSI user label data from Zarr format"""
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_label_user', mode='r')
            
            print("Zarr Array Info:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            print(f"Compressor: {zarr_array.compressors}")
            
            # Convert to numpy array
            data = np.array(zarr_array)
            
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"First 10 values: {data[:10]}")
            print(f"Last 10 values: {data[-10:]}")
            print(f"Unique values: {np.unique(data)}")
            print(f"Value range: {data.min()} to {data.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def load_csi_label_act(self):
        """Load the CSI activity label data from Zarr format"""
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_label_act', mode='r')
            
            print("Zarr Array Info:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            print(f"Compressor: {zarr_array.compressors}")
            
            # Convert to numpy array
            data = np.array(zarr_array)
            
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"First 10 values: {data[:10]}")
            print(f"Last 10 values: {data[-10:]}")
            print(f"Unique values: {np.unique(data)}")
            print(f"Value range: {data.min()} to {data.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def load_csi_label_loc(self):
        """Load the CSI location label data from Zarr format"""
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_label_loc', mode='r')
            
            print("Zarr Array Info:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            print(f"Compressor: {zarr_array.compressors}")
            
            # Convert to numpy array
            data = np.array(zarr_array)
            
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"First 10 values: {data[:10]}")
            print(f"Last 10 values: {data[-10:]}")
            print(f"Unique values: {np.unique(data)}")
            print(f"Value range: {data.min()} to {data.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    
    def load_csi_data_raw(self):
        """Load the CSI data from Zarr format"""
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_data_raw', mode='r')
            
            print("Zarr Array Info:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            print(f"Compressor: {zarr_array.compressors}")
            
            # Convert to numpy array
            data = np.array(zarr_array)
            
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"First 10 values: {data[:10]}")
            print(f"Last 10 values: {data[-10:]}")
            print(f"Unique values: {np.unique(data)}")
            print(f"Value range: {data.min()} to {data.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def load_csi_data_pha(self):
        """Load the CSI data from Zarr format"""
        data = None
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_data_amp', mode='r')
            data = np.array(zarr_array)
        except Exception as e:
            print(f"Error loading data: {e}")
            return data
        
        return data


    def load_csi_data_amp(self):
        data = None
        try:
            # Load the Zarr array
            zarr_array = zarr.open('csi_data_amp', mode='r')
            data = np.array(zarr_array)
        except Exception as e:
            print(f"Error loading data: {e}")
            return data
        return data 


if __name__ == "__main__":
    # Create an instance of the CSI data loader
    loader = CSIDataLoader()
    
    # Load the raw CSI data
    # data = loader.load_csi_data_raw()
    # print(data.shape)
    # print(data.dtype)
    
    # # Load the phase CSI data
    # data = loader.load_csi_data_pha()
    # print(data.shape)
    # print(data.dtype)
    
    # # Load the amplitude CSI data
    # data = loader.load_csi_data_amp()
    # print(data.shape)
    # print(data.dtype)
    
    # Load the location label data
    data = loader.load_csi_label_loc()
    print(data.shape)
    print(data.dtype)
    
    # Load the user label data
    data = loader.load_csi_label_user()
    print(data.shape)
    print(data.dtype)
    
    # Load the activity label data
    data = loader.load_csi_label_act()
    print(data.shape)
    print(data.dtype)
    
    # Load the environment label data
    data = loader.load_csi_label_env()
    print(data.shape)
    print(data.dtype)
    
    