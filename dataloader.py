import torch
import torchvision
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, test_file,root_dir, transform=None):
        train_data = pd.read_csv(train_file)
        le = LabelEncoder()
        train_data['label_encoded'] = le.fit_transform(train_data['label'])
        self.annotations = train_data

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.root_dir + '/' + self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, label)
    
    def plot_image(self, index = None):
        if index is None:
            index = torch.randint(0, len(self), (1,)).item()
        img_path = self.root_dir + '/' + self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        label = self.annotations.iloc[index, 1]
        image = self.transform(image)
        print(index)
        plt.imshow(image)
        plt.title(label)
        plt.show()




if __name__ == "__main__":
    from matplotlib import pyplot as plt

    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    root_dir = 'data'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.RandomRotation(20,expand=True,fill=(255,)),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 0.8)),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.ToTensor()
        ])
    dataset = ImageDataset(train_file, test_file, root_dir, transform)

    dataset.plot_image(7799)
    

    

