from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Type
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class MNIST_dataset:
    '''
    Default Arguments for MNIST dataset
    '''
    ndims: int = 2 # number of dimensions
    inshape: List=field(default_factory=lambda: [28,28]) # shape size
    line_thick: int =3 # size of prior shape


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    '''
    Default Parameters for TEDS-Net architecture
    '''

    # --- TEDS-Net Varibles:
    act: int = 1 # activation function on
    diffeo_int: int = 8  # number of integration layers within network

    # --- Guassian Smoothing Function:
    guas_smooth: int = 1 # include guassian smoothing between composition layers
    Guas_kernel: int = 3 # size of smoothing kernel
    sigma: float=2.0 # guassian sigma

    # --- Upsampling:
    mega_P: int = 2 # how much to upsample the flow field by

    # --- Branches of network:
    dec_depth: List=field(default_factory=lambda: [1]) # smaller input so requires smaller network than in the MICCAI paper


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    '''
    Default Parameters for General Network Architecture
    '''

    dropout: int = 1 # Include dropout
    fi: int = 12 # initial number of feature maps
    net_depth: int=2 # network depth
    in_chan: int=1 # number of channels in
    out_chan: int=1 # number of channel out


@enforce_types
@dataclass(frozen=True)
class LossParams:
    loss: List = field(default_factory=lambda: ['dice','grad'])
    weight: List = field(default_factory=lambda: [1,150])


@enforce_types
@dataclass_json
@dataclass
class Parameters:


    # Training Parameters:
    epoch: int=20
    lr: float = 0.0001
    batch: int = 200
    threshold: float = 0.3

    # tempory data path
    data_path: str="tmp"

    # Loss Parameters:
    loss_params:  LossParams = LossParams()

    # Network Hyper Parameters:
    network_params: GeneralNet = GeneralNet()

    # Default version of network and data selection:
    net: str='teds'
    network: TEDS_Arch = TEDS_Arch()
    data: str='mnist'
    dataset: MNIST_dataset = MNIST_dataset()
    

