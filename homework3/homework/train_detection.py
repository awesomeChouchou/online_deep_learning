# Used AI assistance for conceptual hints and logic structuring.
import torch
import torch.nn.functional as F
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import ConfusionMatrix


def dice_loss(logits, targets, num_classes=3, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train(
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name).to(device)

    train_loader = load_data("drive_data/train", shuffle=True, transform_pipeline="aug", batch_size=batch_size)
    val_loader = load_data("drive_data/val", shuffle=False, transform_pipeline="default", batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)

    # class weights to handle background imbalance
    weights = torch.tensor([1.0, 5.0, 5.0]).to(device)
    loss_fn_ce = torch.nn.CrossEntropyLoss(weight=weights)
    loss_fn_depth = torch.nn.L1Loss()

    print(f"Training {model_name} on {device} | epochs={num_epoch}, lr={lr}, batch={batch_size}")

    best_iou = 0.0

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            target_seg = batch['track'].to(device)
            target_depth = batch['depth'].to(device)

            logits, depth = model(images)

            loss_ce = loss_fn_ce(logits, target_seg)
            loss_dice = dice_loss(logits, target_seg, num_classes=3)
            loss_seg = loss_ce + loss_dice
            loss_depth = loss_fn_depth(depth, target_depth)
            loss = 3.0 * loss_seg + loss_depth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        conf_matrix = ConfusionMatrix(num_classes=3)
        total_mae = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                images = batch['image'].to(device)
                target_seg = batch['track'].to(device)
                target_depth = batch['depth'].to(device)

                logits, depth = model(images)
                pred_seg = logits.argmax(dim=1)
                conf_matrix.add(pred_seg, target_seg)
                total_mae += torch.abs(depth - target_depth).mean().item()

        metrics = conf_matrix.compute()
        miou = metrics['iou']
        avg_mae = total_mae / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"mIoU: {miou:.4f} | MAE: {avg_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if miou > best_iou:
            best_iou = miou
            save_model(model)
            print(f"  -> Saved (best mIoU: {best_iou:.4f})")

    print(f"Done. Best mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    train()
