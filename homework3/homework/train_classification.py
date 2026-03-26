# Used AI assistance for conceptual hints and logic structuring.
import torch
from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric

def train(
    model_name: str = "classifier",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 128,
):
    """
    파라미터를 받아 모델을 학습시키는 메인 함수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드 (models.py의 load_model 활용)
    # **model_kwargs를 통해 추가 인자 전달 가능
    model = load_model(model_name).to(device)
    
   # 3. 데이터 로더 설정
    # 학습용 데이터에는 'train' 증강 파이프라인 적용
    train_loader = load_data("classification_data/train", transform_pipeline="aug", batch_size=batch_size)
    val_loader = load_data("classification_data/val", transform_pipeline="default", batch_size=batch_size)


    # 2. 옵티마이저 및 손실 함수
    # lr 파라미터를 여기서 적용합니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3, # 시작보다 조금 높은 값
    steps_per_epoch=len(train_loader), 
    epochs=num_epoch
    )
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

 
    print(f"Starting training: {model_name} on {device} (Epochs: {num_epoch}, LR: {lr})")

    # 4. 학습 루프
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
          

            train_loss += loss.item()
            optimizer.step()
            scheduler.step()

   

        # 5. 매 에폭마다 검증 수행
        model.eval()
        accuracy = AccuracyMetric()
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model.predict(images)
                accuracy.add(pred, labels)

        current_acc = accuracy.compute()
        #print(f"Epoch {epoch+1:02d}/{num_epoch} | Loss: {train_loss/len(train_loader):.4f} | Acc: {current_acc:.4f}")
        cc_val = current_acc["accuracy"]  # 딕셔너리에서 정확도 값만 추출
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | "
      f"Acc: {current_acc['accuracy']:.4f} (Samples: {int(current_acc['num_samples'])})")
    
    
    # 6. 학습 완료 후 모델 저장
    save_path = save_model(model)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    # 스크립트로 직접 실행할 때의 기본값
    train()