import numpy as np
import raster_geometry as rg
import torch
from torch.utils.data import Dataset
import torchvision


class MNIST_dataclass(Dataset):

    def __init__(self,
        params,
        subset):

        self.params = params
        assert subset in ['Train','Test','Validation']


        if subset =='Train' or subset =='Validation':

            # Download the MNIST dataset:
            mnist_set = torchvision.datasets.MNIST(root=params.data_path, train=True, download=True, transform=None)


            # Seperate into labels and images:
            Y_class= np.array([x[1] for x in mnist_set]) # y class
            Y_set = np.array([np.array(x[0]) for x in mnist_set]) # labels
            self.Y_set =Y_set[np.where(Y_class ==0)[0]]  # only want 0s
            # Divide into Validation and Train:
            N = len(self.Y_set)
            t =int(N*0.1) 

            if subset =='Validation':
                self.X_set = self.Y_set[:t]
                self.Y_set = self.Y_set[:t]

            elif subset=='Train':
                self.X_set = self.Y_set[t:N]
                self.Y_set = self.Y_set[t:N]

        elif subset =='Test':
            mnist_set = torchvision.datasets.MNIST(root=params.data_path, train=False, download=True, transform=None)
            Y_class= np.array([x[1] for x in mnist_set]) # y class
            X_set = np.array([np.array(x[0]) for x in mnist_set]) # labels
            self.X_set =X_set[np.where(Y_class ==0)[0]] # Onlt choose 0s
            self.Y_set = self.X_set

        self.Y_set[self.Y_set>1] =1
        self.Y_set[self.Y_set<1] =0

        # Generate the Prior Shape:
        dim = self.params.dataset.inshape 
        rad=int(dim[0]/4)
        thick=self.params.dataset.line_thick 
        self.ps = rg.circle(dim,radius=rad).astype(int) - rg.circle(dim,radius=(rad-thick)).astype(int) 



    def __len__(self):
        # Return the number of files for subset:train/validation/test
        return len(self.X_set)

    def __getitem__(self, idx):
        """ Genearte the MNIST dataset for each batch

        output:
        x: image
        y_seg: segmentation
        ps: prior shape
        """

        x = np.expand_dims(self.X_set[idx],axis=0)
        y_seg = np.expand_dims(self.Y_set[idx],axis=0)
        ps = np.expand_dims(self.ps,axis=0)

        x = torch.from_numpy(x.astype(np.float32))
        y_seg= torch.from_numpy(y_seg.astype(np.float32))
        ps= torch.from_numpy(ps.astype(np.float32))

        return x,ps, y_seg
    


    def GenPriorShape(self):
        """ Generate Prior Shape Dictionary for each Digit


        """

        dim = self.params.dataset.inshape 

        rad=int(dim[0]/3)
        thick = self.params.dataset.line_thick 
        
        # Solid Circle: b_{0}=1, b_{1}=0: e.g. 1,2,3,4,5,7
        self.T_1 = rg.circle(dim,radius=rad).astype(int)

        # Circle with a hole: b_{0}=1, b_{1}=1: e.g. 0, 6, 9
        rad=int(dim[0]/4)
        self.T_0 = rg.circle(dim,radius=rad).astype(int) - rg.circle(dim,radius=(rad-thick)).astype(int) 

        # Circle with 2 holes: b_{0}=1, b_{1}=2: e.g. 8
        self.T_2 = (rg.circle(dim,radius=rad,position=0.35).astype(int) - rg.circle(dim,radius=(rad-thick),position=0.35).astype(int)) + (rg.circle(dim,radius=rad,position=0.65).astype(int) - rg.circle(dim,radius=(rad-thick),position=0.65).astype(int))
        self.T_2[self.T_2>1] = 1


        self.topo_dict = {0:self.T_0,
                1:self.T_1,
                2:self.T_1, 
                3:self.T_1,
                4:self.T_1,
                5:self.T_1,
                6:self.T_0,
                7:self.T_1,
                8:self.T_2,
                9:self.T_0}

    
    def SelectPrior(self,lab):
        """ Select the prior shape form the shapes -

        """
        return self.topo_dict[lab]


