import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


#訓練圖片前處理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(),#水平翻轉
    transforms.RandomRotation(15),#照片旋轉

    transforms.ColorJitter(


        brightness=0.2,
        contrast=0.2
    ),#照片明暗

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#測試圖片預處理

def main():
    train_dataset = datasets.ImageFolder(
        root="picture/train",
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )



    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

    model = models.mobilenet_v2(weights='IMAGENET1K_V2')


    model.classifier[1] = nn.Linear(
        1280,
        5
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用裝置:", device)

    model = model.to(device)
    print("模型位置:",
          next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    os.makedirs("model", exist_ok=True)
    best_acc = 0
    epochs = 200
    print("開始訓練")
    for epoch in range(epochs):

        print("Epoch開始")

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)


            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss:{running_loss:.3f} "
            f"Accuracy:{acc:.2f}%"
        )

        acc = 100 * correct / total

        if acc > best_acc:
            best_acc = acc

            torch.save(
                model.state_dict(),
                "models/best_mobilenet_v2.pth"
            )
            print("模型儲存完成")

        
    os.makedirs("model", exist_ok=True)



if __name__ == "__main__":
    main()
