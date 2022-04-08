from sklearn.metrics import confusion_matrix
from dataset import ModelNetDataLoader

import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import importlib
import shutil
import hydra
import omegaconf
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, loader, num_class=5):
    mean_correct = []
    confusion_matrix = np.zeros((num_class,num_class))
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0] # labels 
        
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points) # [prediction value] (batch size,5) tensor 
        
        
        pred_choice = pred.data.max(1)[1] # predicted classes     tensor size = batch size
        # calculate confusion matrix
        for i in pred_choice:
            for k in target.cpu():
                confusion_matrix[i,k] += 1
         
         
        
        for cat in np.unique(target.cpu()):       # np.unique :  remove the duplicate numbers and sort 
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum() # the number of correct predictions for class X
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0]) #accuracy for class X
            class_acc[cat,1]+=1 
        correct = pred_choice.eq(target.long().data).cpu().sum() # the number of correct predictions for all classes in one batch
        mean_correct.append(correct.item()/float(points.size()[0])) # accuracy for all classes in one batch
    
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    print(class_acc)
    class_acc = np.mean(class_acc[:,2]) # average of mean accuracy for all classed
    print(class_acc)
    instance_acc = np.mean(mean_correct) # global accuracy 
    print(confusion_matrix)
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('MMA/')

   
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 5
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).to(device)

    try:
        checkpoint = torch.load('best_model.pth')
        
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use existed model')
    except:
        logger.info('No existing model, starting training from scratch...')
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader)

    logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

    
 
    logger.info('end of test...')
    
if __name__ == '__main__':
    main()