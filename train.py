import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from dataloader import get_dataloader
from utils.yaml_operator import read_yaml,write_yaml
from model import MyResNet18
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(config):

    # data
    train_transforms = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40,fill=(255,)),
        # transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_loader, test_loader = get_dataloader(config['batch_size'],train_transforms, test_transforms,num_workers=8,ratio=0.2)
    # logs
    exp_name = config['experiment_name'] + '_' + config['model'] + '_' + config['suffix']
    print(f'Experiment name: {exp_name}')
    os.makedirs(f'logs/{exp_name}', exist_ok=True)
    write_yaml(config, f'logs/{exp_name}/config.yaml')
    writer = SummaryWriter(log_dir="logs/"+exp_name)
    # model
    if config['model'] == 'resnet18':
        model = MyResNet18(class_num=config['num_classes'],pretrained=config['pretrained'])  
    class_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    # device
    if len(config['device_id']) > 0 and torch.cuda.is_available():
        device = [f'cuda:{i}' for i in config['device_id']]
    else:
        device = [torch.device('cpu')]
    print(f'Using {device} device')
    model = model.to(device[0])
    if len(device) > 1:
        print('Using DataParallel')
        model = nn.DataParallel(model, device_ids=device)
    else:
        model = model.to(device[0])

    start_epoch = 0
    if config['resume']:
        checkpoint = torch.load(config['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    
    # train
    for epoch in range(start_epoch,start_epoch+config['epochs']):
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        train_loss = 0.0
        test_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        total = 0
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training Progress")):
            optimizer.zero_grad()
            images = images.to(device[0])
            labels = labels.to(device[0])
            outputs = model(images)
            loss = class_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)
            total += labels.size(0)
        train_loss /= len(train_loader)
        train_acc /= total
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        print(f'Train Loss: {train_loss}, Train Acc: {train_acc}')
        model.eval()
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing Progress")):
                images = images.to(device[0])
                labels = labels.to(device[0])
                outputs = model(images)
                loss = class_loss(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == labels.data)
                total += labels.size(0)
            test_loss /= len(test_loader)
            test_acc /= total
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            print(f'Test Loss: {test_loss}, Test Acc: {test_acc}')
        # save model
        if epoch % config['save_interval'] == 0:
            print(f'Saving model at epoch {epoch}')
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'logs/'+exp_name+f'/model{epoch}.pth')
    pass

if __name__ == '__main__':
    config_file = 'config/leaf_class.yaml'
    config = read_yaml(config_file)
    train(config)