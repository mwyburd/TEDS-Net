import os
import numpy as np
import raster_geometry as rg
import torch
from torch.utils.data import Dataset


class ACDC_dataclass(Dataset):
    ''' ACDC Dataclass

    Using the ACDC dataset, available at: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
    This task only using the myocaridum segmentation (label 2)

    NEW USERS TO-DO:
    1) Add list of IDS Path
    2) Check datapath stored in Parameter file: e.g. params.data_path

    '''

    def __init__(self,
        params,
        subset,
 
    ):
        self.params=params

        assert subset in ['Train','Test']
        
        self.list_IDS = "< LIST OF IDS OF CERTAIN SUBSET >"
   
    
        # --- Generate Prior Shape ---
        rad,thick =  params.dataset.ps_meas
        M,N =  params.dataset.inshape
        self.prior =rg.circle((M,N),radius=rad).astype(int) - rg.circle((M,N),radius=(rad-thick)).astype(int)

    def __len__(self):
        # Return the volumes in that data subet
        return len(self.list_IDS)

    def __getitem__(self, idx):

    
        ID = self.list_IDS[idx]

        # Load in volume and segmentation:
        x =np.load(os.path.join(self.params.data_path,f"Vol/{ID}"))
        y_seg=np.load(os.path.join(self.params.data_path,f"Vol/{ID}"))
        

        # Get Data in Torch Convention:
        x = np.expand_dims(x,axis=0)
        x = torch.from_numpy(x.astype(np.float32))
        y_seg = np.expand_dims(y_seg,axis=0)
        y_seg= torch.from_numpy(y_seg.astype(np.float32))
        prior_shape = np.expand_dims(self.prior,0)
        prior_shape =torch.from_numpy(prior_shape.astype(np.float32))

        return x,prior_shape, y_seg
