import os
import sys
import torch
import numpy as np
import numbers
import math
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision.transforms.functional as FF
from network.UNet import ConvBlock,EncoderBranch,DecoderBranch,BottleNeck



class WholeDiffeoUnit(nn.Module):
    '''
    The diffeo block: 
    1. computes the decoder branch
    2. generates an ndims field
    3. applies diffeo intergretion
    4. applies field to prior shape
    '''

    def __init__(self,params,branch=1):
        super(WholeDiffeoUnit,self).__init__()

        # Get parameters from params:
        self.out_channels =params.network_params.out_chan
        self.ndims  = params.dataset.ndims
        self.viscous = params.network.guas_smooth
        self.act = params.network.act
        self.features = params.network_params.fi
        self.dropout = params.network_params.dropout
        self.net_depth=params.network_params.net_depth
        self.dec_depth = params.network.dec_depth[branch]
        self.inshape = params.dataset.inshape
        self.int_steps = params.network.diffeo_int
        self.Guas_kernel = params.network.Guas_kernel
        self.Guas_P = params.network.sigma
        self.mega_P = params.network.mega_P

        # GET DECODER OUTPUT:
        # happy with this
        self.dec =DecoderBranch(features=self.features,ndims=self.ndims,net_depth=self.net_depth,dec_depth=self.dec_depth,dropout=self.dropout)
    
        
        # Size of initial flow field:
        frac_size_change = [1,2,4,8] # The fractional change
        self.flow_field_size = [int(s/frac_size_change[self.dec_depth-1]) for s in self.inshape]
        # Size of upsampled flow field:
        self.Mega_inshape = [s*self.mega_P for s in self.inshape]

        # 1. GENERATE FIELDS
        self.gen_field =GenDisField(self.dec_depth,self.features,self.ndims)
        
        # 2. Apply diffeomorphic settings :
        self.diffeo_field = DiffeoUnit(self.flow_field_size,self.Mega_inshape,self.int_steps,self.viscous,self.Guas_kernel,self.Guas_P,self.mega_P)
        # 3.  Apply transform to prior:
        self.transformer = mw_SpatialTransformer(self.Mega_inshape)   

    def forward(self,BottleNeck,enc_outputs,prior_shape):
        # Get decoder:
        dec_output = self.dec(BottleNeck,enc_outputs)
        flow_field =self.gen_field(dec_output)
        flow_upsamp = self.diffeo_field(flow_field,self.act,self.viscous,self.ndims)
        sampled = WarpPriorShape(self, prior_shape, flow_upsamp)
        
        return flow_field,flow_upsamp,sampled



class GenDisField(nn.Module):
    '''
    From the output of the U-Net generate the correct sized fields
    
    input: dec output [batch,feature maps,...]
    output: flow_Field [batch,ndims,...], with inialised wieghts
    '''
    def __init__(self,layer_nb,features,ndims):
        super().__init__()

        if ndims ==3:
            from torch.nn import Conv3d as ConvD
        elif ndims==2:
            from torch.nn import Conv2d as ConvD

        dec_features = [1,1,2,4] # number of features from each decoder level
        self.flow_field= ConvD(dec_features[layer_nb-1]*features, out_channels=ndims, kernel_size=1) # Out_channels =3 (x,y,z), could be kerne=3 (??)
        self.flow_field.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_field.weight.shape))
        self.flow_field.bias = nn.Parameter(torch.zeros(self.flow_field.bias.shape))

    def forward(self,CNN_output):
        return self.flow_field(CNN_output)



class DiffeoUnit(nn.Module):
    '''
    Takes in an initial field and ouputs the final upsampled field:

    Takes in an [ndim,M,N] array which acts as the field
    1. Activation Func:
    2. Amplify across integration layers:
    3. Super upsample to quantiity needed:
    ''' 

    def __init__(self, flow_field_size,mega_size,int_steps=7,viscous=1,Guas_kernel=5,Guas_P=2,mega_P=2):
        super(DiffeoUnit,self).__init__()
        
        # --  1. Intergration Layers:
        self.flow_field_size=flow_field_size
        self.integrate_layer = mw_DiffeoLayer(flow_field_size, int_steps,Guas_kernel,Guas_P=Guas_P) 
        
        # -- 2. Mega Upsample:
        self.Mega_inshape = mega_size
        modes = {2:'bilinear',3:'trilinear'}
        self.MEGAsmoothing_upsample =  nn.Upsample(self.Mega_inshape,mode=modes[len(flow_field_size)],align_corners=False)

    def forward(self,flow_field,act,viscous,ndims):

        # 1. Activation Func = between the required amounts:
        if act:
            flow_field= DiffeoActivat(flow_field,self.flow_field_size)

        # 2. Get the displacment field:
        amplified_flow_field = self.integrate_layer(flow_field,viscous)

        # 3. Super Upsample:
        flow_Upsamp= self.MEGAsmoothing_upsample(amplified_flow_field) # Upsample


        return flow_Upsamp
 


class mw_DiffeoLayer(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Adapted from: https://github.com/voxelmorph/voxelmorph

    """

    def __init__(self, inshape, nsteps,kernel=3,Guas_P=2):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        # Set up spatial transformer to intergrate the flow field:
        self.transformer = mw_SpatialTransformer(inshape)

        # ------------------------------
        # SMOOTHING KERNEL:
        # ------------------------------
        ndims = len(inshape)
        self.sigma=Guas_P
        self.SmthKernel = GaussianSmoothing(channels=ndims, kernel_size=kernel, sigma=Guas_P, dim=ndims)
        # ------------------------------
        # ------------------------------

    def forward(self, vec,viscous=1):

        for n in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if viscous:
                # if viscous methods, then smooth at each composition:
                vec =self.SmthKernel(vec)

        return vec



class mw_SpatialTransformer(nn.Module):
    '''
    The pytorch spatial Transformer

    Pytorch transformers require grids generated between -1---1.
    src - Prior shape or flow field [2,3,x,x,x]
    flow - 
    '''
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # create sampling grid (in Pytorch terms)
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid) # not trained by the optimizer, saves memory


    def forward(self,src,flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Pytorch requires axis switch:
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2, 1, 0]] 

            
        return F.grid_sample(src, new_locs, align_corners=True)



class GaussianSmoothing(nn.Module):
    """
    Adrian Sahlman:
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel. If it is less than 0, then it will learn sigma
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size=5, sigma=2, dim=2):
        super(GaussianSmoothing, self).__init__()
        #sigma =2
        self.og_sigma = sigma
        
        kernel_dic = {3:1,5:2}
        self.pad  =kernel_dic[kernel_size]

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                    torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
    
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if self.og_sigma<0:
            # --- Learnable Sigma---------------:
            sigma = 2
            self.learnable = 1 # is the network learnable or static?

            if dim == 1:
                self.conv = nn.Conv1d(in_channels = dim,out_channels = dim,kernel_size =kernel_size,padding=self.pad)
            elif dim == 2:
                self.conv = nn.Conv2d(in_channels = dim,out_channels = dim,kernel_size =kernel_size,padding=self.pad)
            elif dim == 3:
                self.conv = nn.Conv3d(in_channels = dim,out_channels = dim,kernel_size =kernel_size,padding=self.pad)
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

            # Initialse with normal dist
            self.conv.weight = nn.Parameter(torch.cat((kernel,kernel),dim=1))
            self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))


        else:
            # --- Static network---------------:
            self.learnable = 0

            self.register_buffer('weight', kernel)

            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d
            elif dim == 2:
                self.conv = F.conv2d
            elif dim == 3:
                self.conv = F.conv3d
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))


    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # if static or trainable:
        if self.learnable == 1:
            return self.conv(input)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups,padding=self.pad)


def DiffeoActivat(flow_field,size):
    """ Activation Function

    Args:
        flow_field ([tensor array]): A n-dimension array containing the flow feild in each direction
        size ([list]): [description]: The maximum size of the field, to limit the size of the intial displacement

    Returns:
        flow_field [tensor array]: Flow field after the activation funciton has been applied.
    """

    # Assert ndims is 2D or 3D
    assert flow_field.size()[1] in [2,3]
    assert len(size) in [2,3]
    

    if len(size) ==3:
        flow_1= torch.tanh(flow_field[:,0,:,:,:])*(1/size[0]) 
        flow_2 = torch.tanh(flow_field[:,1,:,:,:])*(1/size[1])
        flow_3= torch.tanh(flow_field[:,2,:,:,:])*(1/size[2])
        flow_field =torch.stack((flow_1,flow_2,flow_3), dim=1)
    elif len(size)==2:
        flow_1= torch.tanh(flow_field[:,0,:,:])*(1/size[0])
        flow_2 = torch.tanh(flow_field[:,1,:,:])*(1/size[1])
        flow_field =torch.stack((flow_1,flow_2), dim=1)

    return flow_field



def WarpPriorShape(self, prior_shape, disp_field):
    '''
    Tranform a set of prior shapes:
    '''
    # Apply displacment field
    disp_prior_shape = self.transformer(prior_shape, disp_field)

    return disp_prior_shape



