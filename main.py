from Data.dataloader import AstroDenoisingDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from Models.unet import UNet
from Scripts.train_unet import train_validate_test
from Scripts.red_inference import red_restore

# CONFIG
VERBOSE = True
train = False

if __name__ == "__main__":
    # Set up shared dataset and loaders
    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Train UNet
    model = UNet()
    if train:
        train_validate_test(model, train_loader, val_loader, test_loader)
    else:
        model.load_state_dict(torch.load("output/unet/best_model.pth"))
        if VERBOSE:
            print("Loaded pre-trained UNet model.")

    # RED inference (on the first test image)
    noisy_img, _ = next(iter(test_loader))
    red_restore(model, noisy_img.squeeze(0))