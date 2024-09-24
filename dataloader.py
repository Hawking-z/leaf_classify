import torch
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, df,root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    @property
    def targets(self):
        return self.annotations['label']

    def __getitem__(self, index):
        img_path = self.root_dir + '/' + self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # print(self.transform)
        if self.transform:
            image = self.transform(image)
        return (image, label)


def get_dataloader(batch_size,train_transforms=None, test_transforms=None,ratio=0.2,num_workers=4):

    train_df = pd.read_csv('data/train.csv')
    
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])

    dataset = ImageDataset(df = train_df, root_dir='data')
    
    # 获取数据集的所有索引
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # 获取标签（如果需要分层，可以通过 dataset.targets 获取）
    targets = dataset.targets  

    # 使用 train_test_split 来划分训练集和测试集
    train_indices, test_indices = train_test_split(indices, test_size=ratio, stratify=targets)

    # 使用 Subset 创建新的训练集和测试集
    dataset.transform = train_transforms
    train_dataset = Subset(dataset, train_indices)
    dataset.transform = test_transforms
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return train_loader, test_loader

def get_predict_data_loader(batch_size, num_workers=4):
    train_df = pd.read_csv('data/train.csv')
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])

    test_df = pd.read_csv('data/sample_submission.csv')
    test_df['label'] = 0
    dataset = ImageDataset(df=test_df, root_dir='data')

    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset.transform = test_transforms
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return test_loader,le

if __name__ == '__main__':
    
    train_transforms = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        # transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_loader, test_loader = get_dataloader(64,train_transforms, test_transforms)
    for i,(images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels)
        break
    test_loader,le = get_predict_data_loader(64)

        

    

