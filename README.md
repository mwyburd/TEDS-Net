## TEDS-Net Implementation ##
Overview of architecture and implementation of TEDS-Net, as described in MICCAI 2021: "TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee TopologyPreservation in Segmentations"


Updated code (Jan 2023) now including a brief training script with a mock MNIST dataset, to perform "0" segmentation. A parameter file is also included to describe the hyper-paramters used for the ACDC training and a code for the prior shape is in the dataloader. If using the ACDC example, ensure to ammend the datapaths in both the hyperparameter file and dataloader.


--------------- MOCK EXAMPLE --------------- 

To test the TEDS-Net architecture, I recommend using the mock example, "0" segmentation to ensure everything is installed correctly. 

Running:

>> train_runner.py

will train TEDS-Net for 20 epochs (which shouldn't take more than a minute) and should produce a final test that is similar to:

 >> - - - - - - - - - - - - - - - - - - - ----------------------
 >> Test Dice Loss: 0.9272134661674499 +/- 0.004152349107265217 
 >> - - - - - - - - - - - - - - - - - - - -----------------------

![Alt text](https://github.com/mwyburd/TEDS-Net/blob/main/MNIST_0_Example.png "Mock Example")

and should produce predictions like this
