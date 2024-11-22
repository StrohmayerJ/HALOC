import torch
import torchvision.models as models
import torch.nn as nn
import datasets as data
import argparse
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(opt):
    if torch.cuda.is_available():
        device = "cuda:"+opt.device
    else:
        device = "cpu"

    print("Loading HALOC dataset...")
    # Training data loader
    subsetTrain0 = data.HALOC(opt.data+"/0.csv",windowSize=opt.ws) # train sequence 0
    subsetTrain1 = data.HALOC(opt.data+"/1.csv",windowSize=opt.ws) # train sequence 1
    subsetTrain2 = data.HALOC(opt.data+"/2.csv",windowSize=opt.ws) # train sequence 2
    subsetTrain3 = data.HALOC(opt.data+"/3.csv",windowSize=opt.ws) # train sequence 3
    datasetTrain = torch.utils.data.ConcatDataset([subsetTrain0,subsetTrain1,subsetTrain2,subsetTrain3])
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,batch_size=opt.bs, shuffle=True,num_workers=opt.workers,drop_last=True)
    
    # validation data loader
    datasetVal = data.HALOC(opt.data+"/4.csv",windowSize=opt.ws) # validation sequence
    dataloaderVal = torch.utils.data.DataLoader(datasetVal,batch_size=opt.bs, shuffle=False,num_workers=opt.workers)
    
    # test data loader
    datasetTest = data.HALOC(opt.data+"/5.csv",windowSize=opt.ws) # test sequence
    dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=opt.bs, shuffle=False,num_workers=opt.workers)

    # create dummy resnet18 model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # set number of input channels to 1
    model.fc = nn.Linear(512, 3) # set number of output classes to 3
    model.to(device)

    print("Training...")
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        
        # training loop
        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            feature_window, l = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            prediction = model(feature_window) # TODO: add your model for training here

        # validation loop
        with torch.no_grad():
            for batch in tqdm(dataloaderVal):
                feature_window, l = [x.to(device) for x in batch]
                feature_window = feature_window.float()
                prediction = model(feature_window) # TODO: add your model for validation here

    # test loop
    print("Testing...")
    with torch.no_grad():
        for batch in tqdm(dataloaderTest):
            feature_window, l = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            prediction = model(feature_window) # TODO: add your model for testing here

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/HALOC', help='directory of the HALOC dataset')
    parser.add_argument('--ws', type=int, default=351, help='feature window size (i.e. the number of WiFi packets)')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    train(opt)
    print("Done!")
    torch.cuda.empty_cache()


















