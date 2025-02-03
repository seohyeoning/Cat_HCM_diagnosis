import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import math
from sklearn.metrics import f1_score
from PIL import Image
from torchvision import datasets, models, transforms


# 우리는 처음에 논문에 따라 inceptionNetV3를 사용했고, 상대적으로 크기가 작은 mobilenet과 resnet18에 대해서도  학습을 진행했다. 
# ImageNet으로 사전학습된 모델의 가중치를 불러왔고, dropout(0.5) 추가와 분류기 변경을 적용해서 fine-tuning했다. (분류기만 학습)

    # 방법) pretrain=True 로, feature 추출기의 weight는 freeze, 분류기만 require_grad=True로 설정했다. 
    # 분류기만 학습한 이유: (일반적인 이미지데이터셋에 학습된 모델은,  의료 데이터에서도 유용한 저수준(high-level) 및 고수준(low-level) 특징을 잘 추출할 수 있음.)

# 또한 학습률이 너무 높으면 모델이 초기 에포크에서 빠르게 수렴하면서 검증 손실이 증가할 수 있으므로, 낮은 학습률을 유지했다. (0.001 -> 0.0001)
# Epoch 200에 대해 early stopping을 적용 (PATIENCE=5)하여 과적합을 피하고자 함

# 데이터셋 샘플 수
# Training dataset size: 5278
# Validation dataset size: 530
# 라벨 분포
# Training labels distribution: (array([0, 1]), array([2681, 2597], dtype=int64))
# Validation labels distribution: (array([0, 1]), array([271, 259], dtype=int64))
# from torchvision import transforms
 
 
# 결과 확인: 
# 모바일넷, resnet 각각 epoch 13, 15에서 얼리스타핑되어 모두  Accuracy: 0.5116, F1 Score: 0.3463
# inceptionV3은 아직 28 도는중


# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# Safe image reading
def safe_imread(file_path):
    try:
        stream = open(file_path, "rb")
        bytes_data = bytearray(stream.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[ERROR] Exception in safe_imread for {file_path}: {e}")
        return None



def load_images_with_labels(base_dir):
    images, labels = [], []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = safe_imread(file_path)
                if img is not None:
                    img_resized = cv2.resize(img, (299, 299))
                    images.append(img_resized)

                    # 상위 폴더 이름(HCM, NORMAL)로 레이블 결정
                    if "HCM" in os.path.normpath(root).split(os.sep):
                        labels.append(1)
                    elif "NORMAL" in os.path.normpath(root).split(os.sep):
                        labels.append(0)
                    else:
                        raise ValueError(f"Unexpected folder structure: {root}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, images_np, labels_np):
        self.images = images_np
        self.labels = labels_np

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx].transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class CustomDataset_aug(Dataset):
    def __init__(self, images_np, labels_np, transform=None):
        self.images = images_np
        self.labels = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        
        # PIL로 변환 후 transform 적용
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
# Get dataloaders
def get_dataloaders(train_dir, val_dir, batch_size):
    tr_images, tr_labels = load_images_with_labels(train_dir)
    val_images, val_labels = load_images_with_labels(val_dir)
    train_dataset = CustomDataset(tr_images, tr_labels)
    val_dataset = CustomDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    print(f"Training labels distribution: {np.unique(tr_labels, return_counts=True)}")
    print(f"Validation labels distribution: {np.unique(val_labels, return_counts=True)}")   
    return train_loader, val_loader

def get_dataloaders_aug(train_dir, val_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tr_images, tr_labels = load_images_with_labels(train_dir)
    val_images, val_labels = load_images_with_labels(val_dir)

    train_dataset = CustomDataset(tr_images, tr_labels, transform=train_transforms)
    val_dataset = CustomDataset(val_images, val_labels, transform=val_transforms)
    


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    print(f"Training labels distribution: {np.unique(tr_labels, return_counts=True)}")
    print(f"Validation labels distribution: {np.unique(val_labels, return_counts=True)}")   
    return train_loader, val_loader


# Cosine annealing scheduler
def cosine_annealing(epoch, total_epochs, base_lr):
    return base_lr * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    train_losses, val_losses = [], []
    scaler = GradScaler()
    patience_counter = 0  # Early Stopping counter

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}\n' + '-' * 20)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, correct = 0.0, 0
            all_preds, all_labels = [], []

            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(), torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            epoch_acc = correct.double() / len(train_loader.dataset if phase == 'train' else val_loader.dataset)

            if phase == 'val':
                epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
                val_losses.append(epoch_loss)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

                if epoch_loss < best_loss:
                    print(f"loss감소: {best_loss}-->{epoch_loss}")
                    
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    print("### best model update ###")
                    fin_dir = os.path.join(checkpoint_path, f'best_model_{MODEL_TYPE}.pth')
                    torch.save(best_model_wts, fin_dir)
                    print(f"### 저장 경로: {fin_dir}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    model.load_state_dict(best_model_wts)
                    return model, train_losses, val_losses

            elif phase == 'train':
                train_losses.append(epoch_loss)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                scheduler.step()

    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 16 # 16
    EPOCHS = 100
    BASE_LR = 0.00001
    PATIENCE = 10
    MODEL_TYPE = "mobilenet"  # Choose "mobilenet" or "resnet" or "inceptionnet"
    # all train (전체 param 학습)
        
    train_dir = 'C:/Users/Starlab/Desktop/psh/Cat_HCM/cledd_train_files/train'
    val_dir = 'C:/Users/Starlab/Desktop/psh/Cat_HCM/cledd_train_files/val'
    # checkpoint_path = f'C:/Users/Starlab/Desktop/psh/Cat_HCM/result_{MODEL_TYPE}_2/'
    
    checkpoint_path = f"D:/psh/result_{MODEL_TYPE}_cl_train_lr_e-5"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, BATCH_SIZE)

    if MODEL_TYPE == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        
    elif MODEL_TYPE == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
    
    elif MODEL_TYPE == "inceptionnet":
        model = models.inception_v3(pretrained=True, aux_logits=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose either 'mobilenet' or 'resnet'.")

    model = model.to(device)

    # 이번엔 전체 레이어 학습해보기.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters() if MODEL_TYPE == "mobilenet" else model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, checkpoint_path
    )

    print("Training complete.")
