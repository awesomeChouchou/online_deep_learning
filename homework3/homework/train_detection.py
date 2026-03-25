import torch
from .models import load_model, save_model
from .datasets.road_dataset import load_data
# from .metrics import AccuracyMetric
from .metrics import ConfusionMatrix

def train(
    model_name: str = "detector",
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
    train_loader = load_data("drive_data/train",shuffle=True, transform_pipeline="aug", batch_size=batch_size)
    val_loader = load_data("drive_data/val", shuffle=False, transform_pipeline="default", batch_size=batch_size)


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
    
    loss_fn_seg = torch.nn.CrossEntropyLoss()
    loss_fn_depth = torch.nn.L1Loss() # MAE 계산을 위해 L1Loss 추천

 
    print(f"Starting training: {model_name} on {device} (Epochs: {num_epoch}, LR: {lr})")

    # 4. 학습 루프
    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            target_seg = batch['track'].to(device)# (B, H, W) 형태의 라벨
            target_depth = batch['depth'].to(device)# (B, H, W) 형태의 0~1 값

            #모델 예측
            logits, depth = model(images)
          
            # 각 Loss 계산
            # logits: (B, 3, H, W), target_seg: (B, H, W)
            loss_seg = loss_fn_seg(logits, target_seg)
            
            # depth: (B, H, W), target_depth: (B, H, W)
            loss_depth = loss_fn_depth(depth, target_depth)

            total_loss = loss_seg + 1.0 * loss_depth
            


            #역전파
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += total_loss.item()

   

        # 5. 매 에폭마다 검증 수행
        model.eval()
        conf_matrix = ConfusionMatrix(num_classes=3)
        total_val_mae = 0.0 # Depth MAE 측정을 위해

        with torch.inference_mode():
            for batch in val_loader:
                images = batch['image'].to(device)
                target_seg = batch['track'].to(device)
                target_depth = batch['depth'].to(device)

                logits, depth = model(images) # 모델 통과 추가
                pred_seg = logits.argmax(dim=1)
                conf_matrix.add(pred_seg, target_seg)

                # Depth 평가 (힌트의 MAE < 0.05 달성 확인용)
                total_val_mae += torch.abs(depth - target_depth).mean().item()

        miou = conf_matrix.iou.mean().item()
        avg_mae = total_val_mae / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {total_train_loss/len(train_loader):.4f} | "
              f"mIoU: {miou:.4f} | MAE: {avg_mae:.4f}")
    
    # 6. 학습 완료 후 모델 저장
    save_path = save_model(model)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    # 스크립트로 직접 실행할 때의 기본값
    train()
