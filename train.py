import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from medmnist import INFO, Evaluator
from medmnist.dataset import BreastMNIST, RetinaMNIST, PneumoniaMNIST

from cbam_resnet50 import ResidualNet  # 사용자 정의 CBAM+ResNet

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 연산 일관성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "breastmnist": BreastMNIST,
    "retinamnist": RetinaMNIST,
    "pneumoniamnist": PneumoniaMNIST,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

SAVE_DIR = "./results_cbam"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_and_evaluate(dataset_name):
    print(f"\n📊 Training on {dataset_name.upper()}")
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DatasetClass = DATASETS[dataset_name]

    train_dataset = DatasetClass(split='train', transform=transform, download=True)
    test_dataset = DatasetClass(split='test', transform=transform, download=True)
    in_channels = train_dataset[0][0].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ResidualNet(
        network_type="CIFAR10",
        depth=50,
        num_classes=n_classes,
        att_type="CBAM"
    )
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    log_path = os.path.join(SAVE_DIR, f"{dataset_name}_log.txt")
    with open(log_path, "w") as log_file:

        for epoch in range(50):  # epoch 수 조정 가능
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.squeeze().long().to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            log_file.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # 평가
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.squeeze().long().to(device)
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        evaluator = Evaluator(dataset_name, 'test')
        evaluator.labels = all_labels.numpy()
        metrics = evaluator.evaluate(all_preds)

        log_file.write(f"\nFinal Metrics: {metrics}\n")
        print(f"✅ Evaluation for {dataset_name.upper()}:", metrics)

        # Confusion Matrix
        all_preds = torch.argmax(all_preds, dim=1)

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(info['label'].values()))
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"{dataset_name.upper()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{dataset_name}_confusion_matrix.png"))
        plt.close()

        # 모델 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{dataset_name}_cbam_resnet50.pth"))
        log_file.write(f"Model saved to {dataset_name}_cbam_resnet50.pth\n")

# 전체 실행
if __name__ == "__main__":
    set_seed(42)
    for name in DATASETS:
        train_and_evaluate(name)
