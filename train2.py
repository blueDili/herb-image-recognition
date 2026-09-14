import os
import torch
import torch.nn as nn
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
# 圖片前處理
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




def main():


    # ==========================
    # Dataset
    # ==========================

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



    images,labels = next(iter(train_loader))


    print(images.shape)

    print(labels.shape)



    # ==========================
    # 模型
    # ==========================

    model = HerbCNN(num_classes=5)



    device=torch.device(

        "cuda" if torch.cuda.is_available()
        else "cpu"

    )


    print("使用裝置:",device)



    model=model.to(device)



    print(
        "模型位置:",
        next(model.parameters()).device
    )



    # ==========================
    # Loss / Optimizer
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




    # ==========================
    # Training
    # ==========================

    os.makedirs(
        "model",
        exist_ok=True
    )


    best_acc=0


    epochs=100



    print("開始訓練")



    for epoch in range(epochs):


        model.train()


        running_loss=0

        correct=0

        total=0



        for images,labels in train_loader:


            images=images.to(
                device,
                non_blocking=True
            )


            labels=labels.to(
                device,
                non_blocking=True
            )



            optimizer.zero_grad()



            outputs=model(images)



            loss=criterion(
                outputs,
                labels
            )



            loss.backward()


            optimizer.step()



            running_loss+=loss.item()



            _,predicted=torch.max(
                outputs,
                1
            )


            total+=labels.size(0)


            correct+=(predicted==labels).sum().item()




        acc=100*correct/total



        scheduler.step()



        print(

            f"Epoch {epoch+1}/{epochs} "
            f"Loss:{running_loss:.3f} "
            f"Accuracy:{acc:.2f}%"

        )




        if acc>best_acc:


            best_acc=acc


            torch.save(

                model.state_dict(),

                "models/best_HerbCNN.pth"

            )


            print("模型儲存完成")





if __name__=="__main__":

    main()
