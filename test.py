import warnings
import torch
import torchvision.transforms as T


from deit.models import deit_tiny_distilled_patch16_224
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ImageNetDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None):
        """
        Args:
            huggingface_dataset: Our ImageNet dataset from huggingface
            transform: Potential transformation for the images
        """
        self.dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model = deit_tiny_distilled_patch16_224(pretrained=True)
    print("ok")
    
    imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    imagenet_val_combined = load_dataset('Maysee/tiny-imagenet', split='valid')

    imagenet_val_test = imagenet_val_combined.train_test_split(test_size=0.5, stratify_by_column='label')
    imagenet_val = imagenet_val_test['train']
    imagenet_test = imagenet_val_test['test']

    transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    train_dataset = ImageNetDataset(imagenet_train, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    sample_data = next(iter(train_dataloader))
    print(sample_data)
    print(sample_data.size())
    model(sample_data)
