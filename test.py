import warnings
import torch
import torchvision.transforms as T
import timm

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
    # model = deit_tiny_distilled_patch16_224(pretrained=True)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    print("ok")
    
    imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    imagenet_val_combined = load_dataset('Maysee/tiny-imagenet', split='valid')

    imagenet_val_test = imagenet_val_combined.train_test_split(test_size=0.5, stratify_by_column='label')
    imagenet_val = imagenet_val_test['train']
    imagenet_test = imagenet_val_test['test']
    # imagenet_train = load_dataset('ILSVRC/imagenet-1k', split='train', streaming=True, trust_remote_code=True)
    # imagenet_train = imagenet_train.shuffle(seed=42, buffer_size=1000)

    # Facebook transform
    transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    # google transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), # apparently some images are not RGB
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNetDataset(imagenet_train, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    sample_X, sample_y = next(iter(train_dataloader))
    print(sample_X.size())
    # x, x_dist = model(sample_X)
    x_patched = model.patch_embed(sample_X)
    print(x_patched.size())
    x_emb = model.norm_pre(model.patch_drop(model.pos_drop(x_patched)))
    print(x_emb.size())

    inter = model.forward_intermediates(x_emb)
    print(inter.size())


