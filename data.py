import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# ==========================
# 自架 CNN 模型
# ==========================

class HerbCNN(nn.Module):

    def __init__(self, num_classes=5):
        super(HerbCNN, self).__init__()


        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.5),


            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(256,num_classes)

        )


    def forward(self,x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x





# ==========================
# 測試函式
# ==========================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:


            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            total_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()



    avg_loss = total_loss / len(loader)

    acc = 100 * correct / total


    return avg_loss, acc





# ==========================
# 主程式
# ==========================

def main():


    # ==========================
    # 圖片處理
    # ==========================

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.8,1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),


        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),


        transforms.ToTensor(),


        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )

    ])



    test_transform = transforms.Compose([

        transforms.Resize((224,224)),


        transforms.ToTensor(),


        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )

    ])




    # ==========================
    # Dataset
    # ==========================

    train_dataset = datasets.ImageFolder(

        root="picture/train",

        transform=train_transform

    )


    test_dataset = datasets.ImageFolder(

        root="picture/test",

        transform=test_transform

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=64,

        shuffle=True,

        num_workers=4,

        pin_memory=True

    )



    test_loader = DataLoader(

        test_dataset,

        batch_size=64,

        shuffle=False,

        num_workers=4,

        pin_memory=True

    )



    print("Train數量:",len(train_dataset))

    print("Test數量:",len(test_dataset))





    # ==========================
    # 模型
    # ==========================

    model = HerbCNN(
        num_classes=5
    )



    device=torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    print("使用裝置:",device)



    model=model.to(device)





    # ==========================
    # 訓練設定
    # ==========================

    criterion = nn.CrossEntropyLoss()



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=0.0001,

        weight_decay=1e-4

    )



    scheduler = torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size=30,

        gamma=0.1

    )



    os.makedirs(
        "model",
        exist_ok=True
    )


    best_test_acc = 0


    history = []


    epochs = 100





    # ==========================
    # Training
    # ==========================

    print("開始訓練")



    for epoch in range(epochs):


        model.train()


        running_loss = 0

        correct = 0

        total = 0



        for images, labels in train_loader:


            images = images.to(
                device,
                non_blocking=True
            )


            labels = labels.to(
                device,
                non_blocking=True
            )



            optimizer.zero_grad()



            outputs = model(images)



            loss = criterion(
                outputs,
                labels
            )



            loss.backward()


            optimizer.step()



            running_loss += loss.item()



            _, predicted = torch.max(
                outputs,
                1
            )



            total += labels.size(0)



            correct += (
                predicted == labels
            ).sum().item()




        train_loss = running_loss / len(train_loader)


        train_acc = 100 * correct / total




        test_loss, test_acc = evaluate(

            model,

            test_loader,

            criterion,

            device

        )



        scheduler.step()



        print("======================")

        print(
            f"Epoch {epoch+1}/{epochs}"
        )


        print(
            f"Train Loss:{train_loss:.4f} "
            f"Train Acc:{train_acc:.2f}%"
        )


        print(
            f"Test Loss:{test_loss:.4f} "
            f"Test Acc:{test_acc:.2f}%"
        )




        # Excel紀錄

        history.append({

            "Epoch":epoch+1,

            "Train Loss":train_loss,

            "Train Accuracy":train_acc,

            "Test Loss":test_loss,

            "Test Accuracy":test_acc

        })





        # 儲存最佳模型

        if test_acc > best_test_acc:


            best_test_acc = test_acc


            torch.save(

                model.state_dict(),

                "models/best_HerbCNN.pth"

            )


            print("最佳模型儲存完成")





    # ==========================
    # 輸出 Excel
    # ==========================

    df = pd.DataFrame(history)


    df.to_excel(

        "HerbCNN_training_record.xlsx",

        index=False

    )


    print("======================")

    print("Excel輸出完成")

    print(
        f"最佳Test Accuracy:{best_test_acc:.2f}%"
    )





if __name__=="__main__":

    main()
