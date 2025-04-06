import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import os
import threading

'''
訓練 ResNet50 模型（不使用預訓練權重）
並且使用早期停止（Early Stopping)
訓練過程中會儲存最佳模型（驗證準確率最高的模型）
訓練結果會繪製 Loss 和 Accuracy 曲線圖
'''

# 中文支援
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    train_dir = "../ImageNet/train_t12_ready_100"
    val_dir = "../ImageNet/val_t12_ready_100"
    batch_size = 32 
    num_classes = 100
    num_epochs = 100
    patience = 10
    delta = 0.01

    # 預處理（含資料增強）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
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

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validating"):
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

        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0
            save_path = f"checkpoints/resnet50_best_epoch{best_epoch:02d}_acc{val_acc:.2f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"📦 已儲存最佳模型：{save_path}")
        else:
            early_stop_counter += 1
            print(f"⏸️ 驗證準確率無明顯進步 ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"🛑 提前停止訓練於 Epoch {epoch+1}（最佳為 Epoch {best_epoch}）")
                break
            
            # 啟動計時器，等待使用者輸入
            user_choice = ['y'] 
            input_received = threading.Event()  # 事件，用於標記輸入是否已收到

            def get_input():
                user_input = input("是否繼續訓練？（輸入 'n' 停止，其他繼續）：")
                if user_input.strip().lower() == 'n':
                    user_choice[0] = 'n'
                input_received.set()  # 設置事件，表示輸入已收到

            # 在單獨的線程中執行用戶輸入
            input_thread = threading.Thread(target=get_input)
            input_thread.daemon = True  # 設置為守護線程，主程序退出時不會等待
            input_thread.start()

            # 等待用戶輸入，最多等待 10 秒
            input_received.wait(10)

            if not input_received.is_set():
                print("\n⏱️ 10秒已過，自動繼續訓練...")
            elif user_choice[0] == 'n':
                print("🛑 使用者選擇停止訓練")
                break
            else:
                print("⏩ 繼續訓練...")

    # 繪圖
    print(epoch)
    epochs = list(range(1, epoch+2))
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
    plt.savefig("training_result_plot_resnet50.png")
    print("📈 訓練結果已儲存為 training_result_plot_resnet50.png")

if __name__ == '__main__':
    main()
