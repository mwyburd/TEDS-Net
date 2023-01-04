from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Type
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class ACDC_dataset:
    '''
    Default Arguments for ACDC dataset
    '''
    ndims: int = 2
    inshape: List=field(default_factory=lambda: [144,208])
    ps_meas: List= field(default_factory=lambda: [35,7]) # prior shape measurements
    datapath: str = "<< PATH TO ACDC DATASET >>"
    betti: List=field(default_factory=lambda: [1,1,0,0])


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    '''
    Default Parameters for TEDS-Net architecture
    '''

    # TEDS-Net Varibles:
    act: int = 1 # activation function on
    diffeo_int: int = 8  # number of integration layers within network

    # Guassian Smoothing Function:
    guas_smooth: int = 1 # include guassian smoothing between composition layers
    Guas_kernel: int = 5 # size of smoothing kernel
    sigma: float=2.0 # guassian sigma

    # Upsampling:
    mega_P: int = 2 # how much to upsample the flow field by

    # Branches of network:
    dec_depth: List=field(default_factory=lambda: [4,2]) # The two decoder branches, bulk branch at the 4th layer, FT branch at the 2nd (from top) layer


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    '''
    Default Parameters for General Network Architecture
    '''

    dropout: int = 1 # Include dropout
    fi: int = 12 # initial number of feature maps
    net_depth: int=4 # network depth
    in_chan: int=1 # number of channels in
    out_chan: int=1 # number of channel out


@enforce_types
@dataclass(frozen=True)
class LossParams:
    loss: List = field(default_factory=lambda: ["dice","grad","grad"])
    weight: List = field(default_factory=lambda: [1,10000,10000])


@enforce_types
@dataclass_json
@dataclass
class Parameters:


    # Training Parameters:
    epoch: int=200
    lr: float = 0.0001
    lr_sch: bool=False
    batch: int = 5
    checkpoint_freq: int = 50
    threshold: float = 0.3

    # Loss Parameters:
    loss_params:  LossParams = LossParams()

    # Network Hyper Parameters:
    network_params: GeneralNet = GeneralNet()

    # Dataset
    data: str='ACDC'
    dataset: ACDC_dataset = ACDC_dataset()
    net: str='teds'
    network: TEDS_Arch = TEDS_Arch()


