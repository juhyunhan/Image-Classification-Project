# Ch 02. Image Classification

**흉부 X-ray 이미지로 정상/코로나/폐렴을 분류하는 Image Classification**

1. 이미지 데이터셋 구축
2. Torchvision transforms 라이브러리를 활용한 텐서형 데이터 변환 (모델연산을 위함)
3. Pretrained VGG19 모델을 불러와 Head 부분을 수정
4. Cross entropy Loss function의 이해 및 적용
5. SGDM 최적화기법의 이해 및 적용
6. 인간 추론원리와 닮은 딥러닝 결과의 이해

### Covid chest x-ray 데이터 셋 살펴보기

![Untitled](Ch%2002%20Image%20Classification%201d967a1fa7234adf9af6ddd9efd673ae/Untitled.png)

1. **라이브러리 불러오기**
    
    ```python
    import os
    import copy
    import random
    
    import cv2
    import torch
    import numpy as np
    from torch import nn
    from torchvision import transforms, models
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    from ipywidgets import interact
    
    random_seed = 2022
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```
    
2. **이미지 파일 경로 불러오기**
    
    ```python
    import os
    ```
    
    ```python
    def list_image_files(data_dir, sub_dir):
        image_format = ["jpeg", "jpg", "png"] #이미지 파일 3종류
        
        image_files = [] #이미지 파일을 받아 올 리스트
        images_dir = [os.path.join](https://malwareanalysis.tistory.com/96)(data_dir, sub_dir) 
        for file_path in os.listdir(images_dir): #하위 디렉토리에 있는 모든 이미지를 불러오는 대신, 
            if file_path.split(".")[-1] in image_format: #각 파일의 확장자를 확인을 해서 이미지 포멧에 들어있는 확장자중 하나
                image_files.append(os.path.join(sub_dir, file_path)) #이미지 파일이라는 빈 리스트에 파일경로를 삽입한다. 
        return image_files 
    ```
    
    ```python
    data_dir = "../DATASET/Classification/train/" 
    
    normals_list = list_image_files(data_dir, "Normal") #함수형태로 만들어서 이미지를 쉽게 가져오도록 한다. 
    covids_list = list_image_files(data_dir, "Covid") 
    pneumonias_list = list_image_files(data_dir, "Viral Pneumonia")
    #print를 해서 데이터의 개수가 일치하는지 확인
    ```
    
3. **이미지 파일을 RGB 3차원 배열로 불러오기**
    
    ```python
    import cv2
    ```
    
    ```python
    def get_RGB_image(data_dir, file_name): #data dir와 file path를 연결을 시켜주고,
        image_file = os.path.join(data_dir, file_name) #openCV에서 이미지 형태로 불러들일 수 있게 파일의 경로를 full name으로 만들어준다.
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #openCV 니까 RGB로 변경해준다.
        return image
    ```
    
4. **이미지 데이터 확인하기**
    
    ```python
    import matplotlib.pyplot as plt
    from ipywidgets import interact
    ```
    
    ```python
    min_num_files = min(len(normals_list), len(covids_list), len(pneumonias_list))
    #정상과 코로나의 데이터가 2배 정도 차이 나기 min 함수를 사용한다.
    #print(min_num_files) -> 70
    
    @interact(index=(0, min_num_files-1)) #인터렉티브 하게 보이기 위해서 [데코레이터](https://yaboong.github.io/python/2018/03/04/python-decorator-example/)를 사용한다. / 0부터 69번 인덱스까지 움직일 수 있게 된다.
    def show_samples(index=0): 
        normal_image = get_RGB_image(data_dir, normals_list[index])
        covid_image = get_RGB_image(data_dir, covids_list[index])
        pneumonia_image = get_RGB_image(data_dir, pneumonias_list[index])
        
        plt.figure(figsize=(12, 8))
    
        plt.subplot(131)
        plt.title("Normal")
        plt.imshow(normal_image)
    
        plt.subplot(132)
        plt.title("Covid")
        plt.imshow(covid_image)
    
        plt.subplot(133)
        plt.title("Pneumonia")
        plt.imshow(pneumonia_image)
        plt.tight_layout()
    ```
    
    → 정상과 폐렴 환자의 이미지 구분이 힘들고, 각각의 이미지 크기가 다름
    
5. **학습데이터셋 클래스 구축**
    
    ```python
    train_data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/train/"
    class_list = ["Normal", "Covid", "Viral Pneumonia"]
    ```
    
    dataset을 load 할 수 있는 class 구축, 기본적을 3개의 attribute를 가지게 된다.
    
    1. 생성자 2. dataset의 길이 3. dataset get item
    
    ```python
    class Chest_dataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
    				#하위 폴더에 있는 이미지 파일을 불러온다.
            normals = list_image_files(data_dir, "Normal")
            covids = list_image_files(data_dir, "Covid")
            pneumonias = list_image_files(data_dir, "Viral Pneumonia")
            
    				#파일의 경로는 각 파일의 리스트를 모두 더한 것
            self.files_path = normals + covids + pneumonias
            self.transform = transform
            
        def __len__(self): #데이터 수의 길이를 말한다.
            return len(self.files_path)
        
        def __getitem__(self, index): #인덱스를 호출하게 되면 그에 맞는 이미지와 라벨을 반환 해주게 되는 함수이다.
            image_file = os.path.join(self.data_dir, self.files_path[index]) #해당인덱스에 맞는 파일의 전체 경로를 설정하고, 
            image = cv2.imread(image_file) #openCV의 함수를 이용해서 이미지를 가져온다.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    				#타겟값은 클레스 리스트의 해당하는 인덱스 값으로 받게 된다.
            target = class_list.index(self.files_path[index].split(os.sep)[-2])
             
            target = class_list.index(self.files_path[index].split(os.sep)[0]) #Noraml
            
            if self.transform:
                image = self.transform(image)
                target = torch.Tensor([target]).long() #int값을 변환을 한다.
                
            return {"image":image, "target":target}
    ```
    
    <aside>
    👉 self.변수로 시작하는 것들은 def init 뿐만 아니라 len, getitem 에서도 사용이 가능하고,
    chest dataset으로 호출된 인스턴스로도 활용이 가능하다. |
    
    </aside>
    
    <aside>
    👉 transform은 None으로 두었는데, 향후에 이미지를 텐서형으로 변환하고 또 그 외에 어그멘테이션 기법을 적용할 때 트렌스포머로 인스턴스를 많이 전달하는 편이다.
    
    </aside>
    
    ```python
    dset = Chest_dataset(train_data_dir)
    ```
    
    ```python
    index = 200
    plt.title(class_list[dset[index]["target"]])
    plt.imshow(dset[index]["image"])
    ```
    
6. **배열을 연산가능한 텐서로 변환하기**
    
    ```python
    transformer = transforms.Compose([
        transforms.ToTensor(), #이미지를 텐서로 변경
        transforms.Resize((224, 224)), #224 244 크기로 변환
        transforms.Normalize(mean=[0.5, 0.5, 0.5], #정규화 / 각각 RGB에 해당하는 값이다
                             std=[0.5, 0.5, 0.5])
    ])
    ```
    
    ```python
    train_dset = Chest_dataset(train_data_dir, transformer) #데이터셋에 트랜스포머 객체를 넣고
    ```
    
    ```python
    index = 200
    image = train_dset[index]["image"]
    label = train_dset[index]["target"]
    ```
    
    ```python
    print(image.shape, label)
    ```
    
    torch.Size([3, 224, 224]) 
    
     tensor([2])
    
    ⇒ 텐서형으로 반환된다.
    
7. **데이터로더 구현하기**
    
    데이터로더 함수구현
    
    ```python
    def build_dataloader(train_data_dir, val_data_dir): 
        dataloaders = {}
        train_dset = Chest_dataset(train_data_dir, transformer)
        dataloaders["train"] = DataLoader(train_dset, batch_size=4, shuffle=True, drop_last=True) 
    
        val_dset = Chest_dataset(val_data_dir, transformer)
        dataloaders["val"] = DataLoader(val_dset, batch_size=1, shuffle=False, drop_last=False)
        return dataloaders
    ```
    
    ```python
    train_data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/train/"
    val_data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/test/"
    dataloaders = build_dataloader(train_data_dir, val_data_dir) 
    ```
    
8. **Classification 모델(VGG19) 불러오기**
    
    ![Untitled](Ch%2002%20Image%20Classification%201d967a1fa7234adf9af6ddd9efd673ae/Untitled%201.png)
    
    19의 레이어로 구성된 deep network model
    
    ```python
    model = models.vgg19(pretrained=True)
    ```
    
    ```python
    from torchsummary import summary
    summary(model, (3, 224, 224), batch_size=1, device="cpu")
    #모델아키텍처, (입력 이미지의 체널 수, 높이, 가로), 배치, 디바이스)
    ```
    
    Forward/backward pass size (MB): 238.69
    Params size (MB): 548.05
    Estimated Total Size (MB): 787.31
    
9. **데이터에 맞도록 모델 Head 부분 변경하기**
    
    ```python
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) #지정한 output size로 pooling한다. 
    model.classifier = nn.Sequential( #serial한 연산을 할 수 있도록 묶어주고, model의 classifier라는 함수에 대치한다. 
        nn.Flatten(),
        nn.Linear(512, 256), #fully connected를 위해서 flatten이라는 리니어가 추가된다. 
        nn.ReLU(), #activattion func중 하나인 렐루를 추가한다.
        nn.Dropout(0.1), #overfitting을 위해서 dropout을 사용한다.
        nn.Linear(256, len(class_list)),#256 feature의 사이즈를 클래스의 개수로 지정해서 output shape를 뱉는다.
        nn.Sigmoid() #0-1사이의 값으로 만들어준다. 
    )
    ```
    
    ```python
    def build_vgg19_based_model(device_name='cpu'): #cpu에서 동작 하도록 적음, gpu에서 동작 하도록 할것이면 'cuda'로 적어주면 된다.
        device = torch.device(device_name)
        model = models.vgg19(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(class_list)),
            nn.Softmax(dim=1)
        )
        return model.to(device) #작성된 모델을 cpu로 지정된 torch devic에 !
    ```
    
    ```python
    model = build_vgg19_based_model(device_name='cpu')
    ```
    
    ```python
    from torchsummary import summary
    summary(model, (3, 224, 224), batch_size=1, device="cpu")
    ```
    
    Params size, total size가 많이 줄어든 것을 확인 할 수 있다.
    
10. **손실함수(cross-entropy) 불러오기**
    
    [엔트로피(Entropy)와 크로스 엔트로피(Cross-Entropy)의 쉬운 개념 설명 - Easy is Perfect](https://melonicedlatte.com/machinelearning/2019/12/20/204900.html)
    
    ```python
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    ```
    
    loss 함수를 최소화 하려면, y=1일 때 y pred가 1로 근사해야지 최소화 된다.
    
    y=0일 때는  1- y pred도 1로 근사할 때 최소화 된다. 즉, y pred의 값은 0에 가까울 수록 최소화 된다. 
    
11. **Gradient 최적화 함수 불러오기**
    
    [문과생도 이해하는 딥러닝 (8) - 신경망 학습 최적화](https://sacko.tistory.com/42)
    
    [[NLP]신경망 최적화 함수들 Optimizer: SGD, Momentum, AdaGrad, Adam](https://dokylee.tistory.com/38)
    
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr= 1E-3, momentum=0.9) #SGD(학습가능한 파라미터의 수, 러닝레이트, momentum은 보통 0.9로 설정(원래 설정x)
    ```
    
12. **모델 검증을 위한 Accuracy 생성하기**
    
    ```python
    @torch.no_grad() #backward가 필요없으므로, autograd 형태로
    def get_accuracy(image, target, model):
        batch_size = image.shape[0] #이미지 크기로 batch 크기
        prediction = model(image) #(B, NUM_CLASSES) - [B,0] : 0.1, [B,2] :0.2, [B,2] :0.7
        _, pred_label = torch.max(prediction, dim=1) #(B,1) -> PRED_LABEL : 2 / 가장 높은 confiednce score를 가진 index값을 가져오고, 
        is_correct = (pred_label == target) #index 값과 target 값의 일치 여부 확인 TRUE/FALSE
        return is_correct.cpu().numpy().sum() / batch_size #전체에서 얼마나 빠졌는지 비율로 확인한다.
    ```
    
13. **모델 학습을 위한 함수 구현하기**
    
    ```python
    device = torch.device("cpu")
    ```
    
    ```python
    def train_one_epoch(dataloaders, model, optimizer, loss_func, device): #train을 위한 함수
        losses = {}
        accuracies = {}
        for phase in ["train", "val"]: #한번 이터레이션 할 때 두가지 모두를 시행하게 된다.
            
            running_loss = 0.0 #중간중간 running loss를 받아와서 줄어들고 있는지 확인할 것이다.
            running_correct = 0
            
            if phase == "train":
                model.train()
            else:
                model.eval() 
    				#모델 내부의 batch normalization or dropout 과 같은 train과 val이 다르게 동작하는 레이어 층이 있는데,
    				#레이어들의 기능을 activation 할지 deactivaition 할지 설정하는 것이다.
    				#각각의 로스를 반환하고, overfitting되는지 확인한다.
            
            for index, batch in enumerate(dataloaders[phase]): #dataloader를 phase에 맞게 불러 오고, index값과 batch값에 맞게 가지고 온다.
                image = batch["image"].to(device) #첫번째 리턴 값 : 이미지
                target = batch["target"].squeeze(1).to(device) #2번째 리턴값 : 클래스의 id
                
                optimizer.zero_grad() #미분의 누적값이 누적이되어서 학습에 방해된다. , 함수에 파라미터에 있는 값을 플러시 해준다., 다시 갱신
    
                with torch.set_grad_enabled(phase == "train"): #set_grad_enabled는 내부의 인자 값이 true일 때만 activate
                    prediction = model(image)
                    loss = loss_func(prediction, target)
                    
                    if phase == "train":
                        loss.backward() 
                        optimizer.step()
                
                running_loss += loss.item() #나온 로스 값은 value만 받아서 running_loss에 누적시킨다.
                running_correct += get_accuracy(image, target, model)
                
                if phase == "train":
                    if index % 10 == 0:
                        print(f"{index}/{len(dataloaders[phase])} - Running Loss: {loss.item()}")
    
            losses[phase] = running_loss / len(dataloaders[phase]) 
            accuracies[phase] = running_correct / len(dataloaders[phase])
        return losses, accuracies
    ```
    
    <aside>
    👉 loss, accuracy 값을 저장하고, 반환 
    리스트, 키값으로 append를 시킨다!
    
    </aside>
    
14. **모델 학습 수행하기**
    
    ```python
    device = torch.device("cuda")
    
    train_data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/train/"
    val_data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/test/"
    
    dataloaders = build_dataloader(train_data_dir, val_data_dir)
    model = build_vgg19_based_model(device='cuda')
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr= 1E-3, momentum=0.9)
    ```
    
    ```python
    def train_one_epoch(dataloaders, model, optimizer, loss_function, device):
        losses = {}
        accuracies = {}
        
        for phase in ["train", "val"]:
            running_loss = 0.0
            running_correct = 0
            
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            for index, batch in enumerate(dataloaders[phase]):
                image = batch[0].to(device)
                target = batch[1].squeeze(dim=1).to(device)
    
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(image)
                    loss = loss_func(prediction, target)
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item()
                running_correct += get_accuracy(image, target, model)
                
                if phase == "train":
                    if index % 10 == 0:
                        print(f"{index}/{len(dataloaders['train'])} - Running loss: {loss.item()}")
                
            losses[phase] = running_loss / len(dataloaders[phase])
            accuracies[phase] = running_correct / len(dataloaders[phase])
        return losses, accuracies
    ```
    
    ```python
    num_epochs = 10
    
    best_acc = 0.0
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    
    for epoch in range(num_epochs):
        losses, accuracies = train_one_epoch(dataloaders, model, optimizer, loss_func, device)
        train_loss.append(losses["train"])
        val_loss.append(losses["val"])
        train_accuracy.append(accuracies["train"])
        val_accuracy.append(accuracies["val"])
        
        print(f"{epoch+1}/{num_epochs}-Train Loss: {losses['train']}, Val Loss: {losses['val']}")
        print(f"{epoch+1}/{num_epochs}-Train Acc: {accuracies['train']}, Val Acc: {accuracies['val']}")
        
        if (epoch > 3) and (accuracies["val"] > best_acc):
            best_acc = accuracies["val"]
            best_model = copy.deepcopy(model.state_dict())
            save_best_model(best_model, f"model_{epoch+1:02d}.pth")
            
    print(f"Best Accuracy: {best_acc}")
    ```
    
    ```python
    plt.figure(figsize=(6, 5))
    plt.subplot(211)
    plt.plot(train_loss, label="train")
    plt.plot(val_loss,  label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid("on")
    plt.legend()
    plt.subplot(212)
    plt.plot(train_accuracy, label="train")
    plt.plot(val_accuracy, label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid("on")
    plt.legend()
    plt.tight_layout()
    ```
    
15. **테스트 이미지를 통한 학습 모델 분류성능검증하기**
    
    ```python
    data_data = "../DATASET/Classification/test/"
    class_list = ["Normal", "Covid", "Viral Pneumonia"]
    
    test_normals_list = list_image_files(data_dir, "Normal")
    test_covids_list = list_image_files(data_dir, "Covid")
    test_pneumonias_list = list_image_files(data_dir, "Viral Pneumonia")
    ```
    
    ```python
    def preprocess_image(image):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 244)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
        ]) 
        
        tensor_image = transformer(image)  # (C, H, W)
        tensor_image = tensor_image.unsqueeze(0) # (B, C, H, W)
        return tensor_image
    ```
    
    ```python
    @torch.no_grad()
    def model_predict(image, model):
        tensor_image = preprocess_image(image)  #tensor 변환된 이미지를 사용
        prediction = model(tensor_image)
        
        _, pred_label = torch.max(prediction.detach(), dim=1) #(B, NUM_CLASS)
        pred_label = pred_label.squeeze(0) #dummy batch 지우기 (B,1) -> (1)
        return pred_label.item() #TORCH 변수의 수치적인 값만 갖고 오겠다는 것
    ```
    
    ```python
    ckpt = torch.load("../../cv-project/trained_model/model_06.pth")
    
    model = build_vgg19_based_model(device='cuda')
    model.load_state_dict(ckpt)
    model.eval()
    ```
    
    모델 LOAD
    
    ```python
    min_num_files = min(len(test_normals_list), len(test_covids_list), len(test_pneumonias_list))
    #한번에 보기 위해서 가장 작은수의 이미지를 기준으로 클립
    @interact(index=(0, min_num_files-1))
    def show_result(index=0):
        normal_image = get_RGB_image(data_dir, test_normals_list[index])
        covid_image = get_RGB_image(data_dir, test_covids_list[index])
        pneumonia_image = get_RGB_image(data_dir, test_pneumonias_list[index])
        
        prediction_1 = model_predict(normal_image, model)
        prediction_2 = model_predict(covid_image, model)
        prediction_3 = model_predict(pneumonia_image, model)
        
        plt.figure(figsize=(12,8))
        plt.subplot(131)
        plt.title(f"Pred:{class_list[prediction_1]} | GT:Normal")
        plt.imshow(normal_image)
    
        plt.subplot(132)
        plt.title(f"Pred:{class_list[prediction_2]} | GT:Covid")
        plt.imshow(covid_image)
    
        plt.subplot(133)
        plt.title(f"Pred:{class_list[prediction_3]} | GT:Pneumonia")
        plt.imshow(pneumonia_image)
        plt.tight_layout()
    ```
    
    ```python
    data_dir = "../../cv-project/MEDICAL-DATASET-001/Classification/test/"
    class_list = ["Normal", "Covid", "Viral Pneumonia"]
    
    test_normals_list = list_image_files(data_dir, "Normal")
    test_covids_list = list_image_files(data_dir, "Covid")
    test_pneumonias_list = list_image_files(data_dir, "Viral Pneumonia")
    
    def get_RGB_image(data_dir, file_name):
        image_file = os.path.join(data_dir, file_name)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    ```
    
    ```python
    def preprocess_image(image):
        ori_H, ori_W = image.shape[:2]
        
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])
        
        tensor_image = transformer(image)
        tensor_image = tensor_image.unsqueeze(0)
        return tensor_image
    ```
    
    ```python
    ckpt = torch.load("./trained_model/model_06.pth")
    
    model = build_vgg19_based_model(device_name="cuda")
    model.load_state_dict(ckpt)
    model.eval()
    
    def test_model(image, model):
        tensor_image = preprocess_image(image)
    
        with torch.no_grad():
            prediction = model(tensor_image)
    
        _, pred_label = torch.max(prediction.detach(), dim=1)
        pred_label = pred_label.squeeze(0)
        return pred_label.item()
    ```
    
    ```python
    min_num_files = min(len(test_normals_list), len(test_covids_list), len(test_pneumonias_list))
    
    @interact(index=(0, min_num_files-1))
    def show_samples(index=0):
        normal_image = get_RGB_image(data_dir, test_normals_list[index])
        covid_image = get_RGB_image(data_dir, test_covids_list[index])
        pneumonia_image = get_RGB_image(data_dir, test_pneumonias_list[index])
        
        prediction_1 = test_model(normal_image, model)
        prediction_2 = test_model(covid_image, model)
        prediction_3 = test_model(pneumonia_image, model)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(131)
        plt.title(f"Pred:{class_list[prediction_1]} | GT:Normal")
        plt.imshow(normal_image)
        plt.subplot(132)
        plt.title(f"Pred:{class_list[prediction_2]} | GT:Covid")
        plt.imshow(covid_image)
        plt.subplot(133)
        plt.title(f"Pred:{class_list[prediction_3]} | GT:Viral Pneumonia")
        plt.imshow(pneumonia_image)
        plt.tight_layout()
    ```
    
    정상/폐렴 ACCURACY 혼동