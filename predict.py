import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from torchvision import transforms 
from dataloader import get_predict_data_loader
from utils.yaml_operator import read_yaml
from model import MyResNet18
import pandas as pd


def predict(config):
    if config['model'] == 'resnet18':
        print('Using ResNet18')
        model = MyResNet18(class_num=config['num_classes'],pretrained=config['pretrained'])
    print(f'Loading model from {config["checkpoint_path"]}')  
    checkpoint = torch.load(config['checkpoint_path'])
    # 去掉前缀并加载
    state_dict = checkpoint['model_state_dict']
    if 'module.' in next(iter(state_dict)):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)
    model.eval()
    predict_loader,le = get_predict_data_loader(config['batch_size'])

    result = []
    for i, (images, _) in enumerate(tqdm(predict_loader, desc="predict Progress")):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        result.extend(predicted.cpu().numpy())
    result = le.inverse_transform(result)

    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = result
    submission.to_csv('submission.csv', index=False)
    

if __name__ == '__main__':
    config = read_yaml('config/leaf_class.yaml')
    predict(config)

