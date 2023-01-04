## TEDS-Net Implementation ##
Overview of architecture and implementation of TEDS-Net, as described in MICCAI 2021: "TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee TopologyPreservation in Segmentations"


Updated code (Jan 2023) now including a brief training script with a mock MNIST dataset, to perform "0" segmentation. 

--- MOCK EXAMPLE ---

To test the TEDS-Net architecture, I recommend using the mock example, "0" segmentation to ensure everything is installed correctly. 

Running:

train_runner.py

will train TED-Net for 20 epochs and should produce a final test that is similar to:

 - - - - - - - - - - - - - - - - - - - -
 Test Dice Loss: 0.9272134661674499 +/- 0.004152349107265217 
 - - - - - - - - - - - - - - - - - - - -

![Alt text](https://github.com/mwyburd/TEDS-Net/blob/main/MNIST_0_Example.png "Mock Example")
