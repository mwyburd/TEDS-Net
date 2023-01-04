import torch
import os
import numpy as np
from tqdm import tqdm
from utils.losses import dice_loss,grad_loss

class Trainer:

    def __init__(self, params, device, net):

        """ Set up the training and evaluations 

        Args:
            params (dict): parameters for training
            device (device): describes what to train the network on
            net (class):  TEDS-Net architecture
        """

        self.params = params
        self.device = device
        self.net = net    

        # load in dataloader
        self.get_dataloader()
        # set up optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.lr)
    

    def dothetraining(self):
        
        # Loop through epochs:
        for epoch in range(self.params.epoch):

            # Perform the Training:
            self.net.train()            
            subset = 'train'
            self.epoch_loss =[]
            for (x,prior_shape, labels) in tqdm(self.dataloader_dic[subset]):

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    # Input into network:
                    output = self.net(x.to(self.device),prior_shape.to(self.device))

                    # Compute losses:
                    loss = self.perform_losses(labels.to(self.device),output)

                    # Update model:
                    loss.backward()
                    self.optimizer.step()

            # Get epoch train loss:
            train_loss=np.mean(self.epoch_loss)

            # Perform the Validation:
            val_loss=self.do_validation(epoch)    
            
            # Print out losses
            print("[{0}] {1}: {2:.6f}".format(epoch, 'training_loss', train_loss))
            print("[{0}] {1}: {2:.6f}".format(epoch, 'validation_loss', val_loss))

    def do_validation(self,epoch):
        """ Perform the validation on each batch:

        """
        self.net.eval()
        self.epoch_loss=[]
        subset = 'validation'
        for (x,prior_shape, labels) in tqdm(self.dataloader_dic[subset]):
            
            with torch.set_grad_enabled(False):

                # Input into network:
                output = self.net(x.to(self.device),prior_shape.to(self.device))

                # Compute the losses:
                self.perform_losses(labels.to(self.device),output)
        
        return np.mean(self.epoch_loss)
        
    def get_dataloader(self):
        """Load in the dataloader:

        """

        if self.params.data=="mnist":
            from dataloaders.setup import setup_mnist_dataloader as setup_dataloader
        elif self.params.data=="ACDC":
            from dataloaders.setup import setup_acdc_dataloader as setup_dataloader

        self.dataloader_dic = setup_dataloader(self.params,['train','validation','test'])

    def perform_losses(self,labels,output):
        """ Perform the losses:
        
        Args:
            labels (tensor): the ground truth label
            output (list): output[0] the prediction, output[1] bulk field, output[2] ft field
        """        

        loss =0
        for i,(loss_function,w) in enumerate(zip(self.params.loss_params.loss, self.params.loss_params.weight)):
            
            # Perform the field regulisation and Dice loss functions on the the outputs
            if loss_function=="dice":
                curr_loss = dice_loss().loss(labels,output[i],loss_mult=w)
            elif "grad" in loss_function:
                curr_loss = grad_loss(self.params).loss(labels,output[i],loss_mult=w)
      
            # Combine our loss functions
            loss += curr_loss

        # Add to losses:
        self.epoch_loss.append(loss.item())

        return loss


    def do_evalutation(self):
        """ Evaluate the Trained network and compute the average Dice Score:

        """

        self.params.batch=1
        test_dice = []
        for (x,prior_shape, labels) in self.dataloader_dic['test']:

            # -- Input into network --------------:
            output = self.net(x.to(self.device),prior_shape.to(self.device))
            output = list(output)
            output[0]=(output[0]>self.params.threshold).int()

            # -- Perform Dice Loss --------------:
            test_dice.append(dice_loss().np_loss(labels.to(self.device),output[0]))

        print(" - -"*10)
        print(f" Test Dice Loss: {1-np.mean(test_dice)} +/- {np.std(test_dice)} ")
        print(" - -"*10)
        self.ViewPrediction(x,labels,prior_shape,output)

    def ViewPrediction(self,x,labels,prior_shape,output):
        """
        
        """
        import matplotlib.pyplot as plt

        # Get a single prediction:
        x =x[0,0,:,:].cpu().numpy().astype(int) # image
        y = labels[0,0,:,:].cpu().numpy().astype(int) # ground truth
        y_hat =output[0][0,0,:,:].cpu().numpy().astype(int) # prediction
        p = prior_shape[0,0,:,:].cpu().numpy().astype(int)# prior

        # make our figure
        fig,ax = plt.subplots(ncols=3)
        cmaps = 'winter','autumn','summer'
        for i,(a,seg,t) in enumerate(zip(ax,[y,p,y_hat],['Label','Prior',"Prediction"])):
            a.imshow(x,cmap='gray')
            mask_lab = np.ma.masked_array(seg, seg == 0)# mask out label
            a.imshow(mask_lab, cmap=cmaps[i], alpha = 0.6)
            a.set_title(t)
            a.axis('off')

        plt.savefig(os.path.join(self.params.data_path,'figure'))

