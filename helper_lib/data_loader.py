import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data_loader(
    data_dir,
    batch_size=32,
    split="train",
    dataset_name="CIFAR10",
    transform=None,
    val_ratio=0.1,
    seed=42
):
    if transform is None:
        transform = transforms.ToTensor()

    dataset_name = dataset_name.lower()
    split = split.lower().strip()

    if dataset_name == "cifar10":
        if split in ["train", "val"]:
            full_train_dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            # Create validation split
            val_size = int(len(full_train_dataset) * val_ratio)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
            )

            if split == "train":
                dataset = train_dataset
            else:
                dataset = val_dataset

        elif split == "test":
            
            dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )

        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    elif dataset_name == "mnist":
        if split in ["train", "val"]:
            full_train_dataset = datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            # Create validation split
            val_size = int(len(full_train_dataset) * val_ratio)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
            )

            if split == "train":
                dataset = train_dataset
            else:
                dataset = val_dataset

        elif split == "test":
            
            dataset = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )

        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    else:
        raise ValueError(
            "Unsupported dataset. Choose one of: CIFAR10, MNIST"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train")
    )

    return loader