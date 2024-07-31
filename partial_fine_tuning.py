import sys 
from pathlib import Path

import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

sys.path.append(str(Path(__file__).parent.absolute()))
from trainer.adversarial_trainer import Trainer
from model.adversarial_simsiam import SimSiam
from dataloader.adversarial_simsiam_dataset import AdversarialSimSiamDataset
from dataloader.multi_view_dataloader import MultiViewDataset
from preprocess.base_augmentation import get_eval_transform
from preprocess.semi_supervised_augmentation import get_train_transform

def main():
    
    file_name = Path(__file__).stem
    dataset_name = "with_deblur"
    
    # --- Setting for training --- #
    epochs = 1000
    batch_size = 32
    init_lr = 0.1 * batch_size/32
    weight_decay = 1.5e-6
    
    log_dir = Path(f"./logs/{file_name}/{dataset_name}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_dir = str(log_dir)
    
    checkpoint_dir = Path(f"./checkpoint/{file_name}/{dataset_name}")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    checkpoint_dir = str(checkpoint_dir)
    
    imagenet_pretrain_path = "./weight/simsiam.pth.tar"
    freeze_partial_param = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Prepare datset --- #
    
    train_dataset_path = f"./dataset/{dataset_name}/train"
    eval_dataset_path = f"./dataset/{dataset_name}/eval"
    test_dataset_path = f"./dataset/{dataset_name}/eval_query"
    validation_dataset_path = f"./dataset/{dataset_name}/validation"
    
    train_transform = get_train_transform()
    eval_transform = get_eval_transform()
    
    eval_image_folder = ImageFolder(eval_dataset_path, Compose(eval_transform))
    test_image_folder = ImageFolder(test_dataset_path, Compose(eval_transform))
    train_dataset = AdversarialSimSiamDataset(train_dataset_path, train_transform)
    validation_dataset = AdversarialSimSiamDataset(validation_dataset_path, train_transform)
    
    test_dataloader = DataLoader(test_image_folder, batch_size=batch_size, shuffle=False, num_workers=8)
    eval_dataloader = DataLoader(eval_image_folder, batch_size = batch_size, shuffle=False, num_workers=8)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    # --- Prepare model --- #
    simsiam = SimSiam(backbone="res50", 
                      backbone_dim=2048, 
                      prediction_dim=512, 
                      num_classes=len(train_dataset.classes),
                      imagenet_pretrain_path=imagenet_pretrain_path,
                      freeze_partial_param=freeze_partial_param
                      )
    
    # --- Trainer ---#
    trainer = Trainer(simsiam, 
                      train_dataloader, 
                      validation_dataloader,
                      eval_dataloader,
                      test_dataloader, 
                      num_epochs=epochs,
                      init_lr=init_lr,
                      weight_decay=weight_decay,
                      log_dir=log_dir,
                      checkpoint_dir=checkpoint_dir,
                      device=device, 
                      name=file_name
                      )
    
    trainer.run()
    
if __name__ == "__main__":
    main()