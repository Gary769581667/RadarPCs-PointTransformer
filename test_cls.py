import numpy as np
from tqdm import tqdm
import torch
from dataset import ModelNetDataLoader
import importlib
import shutil
import hydra
import omegaconf
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test(model, loader, num_class=5):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    '''HYPER PARAMETER'''
   
    
    DATA_PATH = hydra.utils.to_absolute_path('MMA/')
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.num_class = 5
    args.input_dim = 6

    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).to(device)
    try:
        checkpoint = torch.load('best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("no trained module found")
    # print(classifier)
    with torch.no_grad():
        instance_acc, class_acc = test(model = classifier.eval(), loader = testDataLoader)
    print(instance_acc)
    print(class_acc)

if __name__ == '__main__':
    main()