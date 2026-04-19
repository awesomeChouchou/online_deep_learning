# Used AI assistance for conceptual hints and logic structuring.
"""
Usage:
    python3 -m homework.train_planner --model mlp_planner
    python3 -m homework.train_planner --model transformer_planner
    python3 -m homework.train_planner --model cnn_planner
"""

import argparse

import torch
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name).to(device)

    # CNN planner uses images, others use state_only
    if model_name == "cnn_planner":
        train_pipeline = "default"
        val_pipeline = "default"
    else:
        train_pipeline = "state_only"
        val_pipeline = "state_only"

    train_loader = load_data("drive_data/train", shuffle=True, transform_pipeline=train_pipeline, batch_size=batch_size)
    val_loader = load_data("drive_data/val", shuffle=False, transform_pipeline=val_pipeline, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)

    loss_fn = torch.nn.L1Loss(reduction='none')

    print(f"Training {model_name} on {device} | epochs={num_epoch}, lr={lr}, batch={batch_size}")

    best_error = float('inf')

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            if model_name == "cnn_planner":
                pred = model(image=batch['image'].to(device))
            else:
                pred = model(
                    track_left=batch['track_left'].to(device),
                    track_right=batch['track_right'].to(device),
                )

            target = batch['waypoints'].to(device)
            mask = batch['waypoints_mask'].to(device)

            # masked L1 loss
            loss = loss_fn(pred, target)  # (b, n, 2)
            loss = (loss * mask[..., None]).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        metric = PlannerMetric()

        with torch.inference_mode():
            for batch in val_loader:
                if model_name == "cnn_planner":
                    pred = model(image=batch['image'].to(device))
                else:
                    pred = model(
                        track_left=batch['track_left'].to(device),
                        track_right=batch['track_right'].to(device),
                    )

                target = batch['waypoints'].to(device)
                mask = batch['waypoints_mask'].to(device)
                metric.add(pred, target, mask)

        results = metric.compute()
        long_err = results['longitudinal_error']
        lat_err = results['lateral_error']
        l1_err = results['l1_error']

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Long: {long_err:.4f} | Lat: {lat_err:.4f} | L1: {l1_err:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if l1_err < best_error:
            best_error = l1_err
            save_model(model)
            print(f"  -> Saved (best L1: {best_error:.4f})")

    print(f"Done. Best L1: {best_error:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp_planner",
                        choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train(model_name=args.model, num_epoch=args.epochs, lr=args.lr, batch_size=args.batch_size)
