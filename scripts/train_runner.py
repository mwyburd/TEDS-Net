import torch
import argparse
from trainer import Trainer

class Train_Runner:

    def __init__(self,args):

        # 1) Setup parameters ----------- :
        self.setup_params(args)

        # 2) Setup Device ----------- :
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

        # 3) Load in Model ----------- :
        from network.TEDS_Net import TEDS_Net as net
        net = net(self.params)
        net.to(device)

        # 4) Train and Evalte the Model ---------- :
        trainer = Trainer(self.params, device, net)
        trainer.dothetraining()
        trainer.do_evalutation()


    def setup_params(self,args):
        """ _set up the training parameters from default options provided_

        Args:
            args (module): contains the settings for training
        """
        
        if args.dataset =="mnist":
            from parameters.mnist_parameters import Parameters
        elif args.dataset=="ACDC":
            from parameters.acdc_parameters import Parameters

        self.params = Parameters.from_dict({'data':args.dataset})


if __name__ == '__main__':
    """ MKWyburd GitHub TEDS-Net
    
    The network described in MICCAI 2021 for myocaridum segmentation using the ACDC dataset are stored in parameters/acdc_parameters. The prior shape generator is in the dataloader folder

    To test the code, you can use the simple MNIST example, using Pytorch automated MNIST dataset. The data will be downloaded into a "tmp" folder. 
    """

    parser = argparse.ArgumentParser(
        description='Run TEDS-Net Segmentation')

    parser.add_argument('--dataset', 
                        help = 'Which dataset we are using',
                        choices=['ACDC','mnist'],
                        default='mnist')

    args = parser.parse_args()

    # Perform our Training:
    Train_Runner(args)