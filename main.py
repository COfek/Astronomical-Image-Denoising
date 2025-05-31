from Data.dataloader import AstroDenoisingDataset
from torch.utils.data import DataLoader, random_split

dataset = AstroDenoisingDataset("astro_simulated_dataset_all/clean",
                                "astro_simulated_dataset_all/noisy")

train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")