import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import os

'''
訓練 VGG16 模型（不使用預訓練權重）
並且使用早期停止（Early Stopping)
訓練過程中會儲存最佳模型（驗證準確率最高的模型）
訓練結果會繪製 Loss 和 Accuracy 曲線圖
'''

# 設定中文字型
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    train_dir = "../ImageNet/train_t12_ready_100"
    val_dir = "../ImageNet/val_t12_ready_100"
    batch_size = 16
    num_classes = 100 # 類別數量
    num_epochs = 100 # 訓練的 epoch 數量
    patience = 10 # 提前停止的耐心次數(early stopping patience)
    delta = 0.01  # 驗證準確率的最小提升量

    # 預處理（resize + normalization）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 隨機水平翻轉
        transforms.RandomVerticalFlip(), # 隨機垂直翻轉
        transforms.RandomRotation(10), # 隨機旋轉
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 隨機顏色變化(模擬光線不同)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 建立資料集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)  # 替換分類層
    model = model.to(device)

    # 訓練參數
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    os.makedirs("checkpoints", exist_ok=True)

    # 訓練迴圈
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 驗證
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Validatoin Accuracy: {val_acc:.2f}%")

        # 儲存最佳模型
        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0
            save_path = f"checkpoints/vgg16_best_epoch{best_epoch:02d}_acc{val_acc:.2f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"📦 已儲存最佳模型：{save_path}")
        else:
            early_stop_counter += 1
            print(f"⏸️ 驗證準確率無明顯進步 ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"🛑 提前停止訓練於 Epoch {epoch+1}（最佳為 Epoch {best_epoch}）")
                break

    # 繪製訓練結果圖
    epochs = list(range(1, num_epochs+1))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='訓練 Loss')
    plt.plot(epochs, val_losses, label='驗證 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss 曲線')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='訓練準確率')
    plt.plot(epochs, val_accuracies, label='驗證準確率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('準確率曲線')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_result_plot.png")
    print("📈 訓練結果已儲存為 training_result_plot.png")

if __name__ == '__main__':
    main()
