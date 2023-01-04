
from torch.utils.data import DataLoader    


def setup_mnist_dataloader(params,subset_list):

    from dataloaders.mnist import MNIST_dataclass as MyDataset

    # --------- Dataloaders Functions:
    params_train = {'batch_size': params.batch, 'shuffle': True,
                    'num_workers': 8}

    params_val = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  

    params_test = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  


    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params,subset='Train')
        dataloader_dict['train'] = DataLoader(training_set, **params_train)
    if 'validation' in subset_list:
        val_set = MyDataset(params,subset='Validation')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params,subset='Test')
        dataloader_dict['test'] =DataLoader(test_set, **params_test)


    return dataloader_dict


def setup_acdc_dataloader(params,subset_list):    

    from dataloaders.ACDC import ACDC_dataclass as MyDataset

    # --------- Dataloaders Functions:
    params_train = {'batch_size': params.batch, 'shuffle': True,
                    'num_workers': 8}

    params_val = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  

    params_test = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  


    # Put in my dataloader functions:
    dataset_dict ="<AMEND WITH YOUR DATALOADER DICTIONARY>"

    # Into the torch dataloader function
    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params,dataset_dict['train'],subset='Train',aug=True)
        dataloader_dict['train'] = DataLoader(training_set, **params_train)            

    if 'validation' in subset_list:
        val_set = MyDataset(params,dataset_dict['val'],subset='Train')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params,dataset_dict['test'],subset='Test')
        dataloader_dict['test'] =DataLoader(test_set, **params_test)

    return dataloader_dict