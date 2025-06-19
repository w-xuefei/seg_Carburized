'''
日期: 2025-01-16
作者: wxf

# conda env: base

# 查看训练过程:
tensorboard --logdir tb_logs
http://localhost:6006

'''
import os
import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import yaml
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

import torch
torch.set_float32_matmul_precision('high')  # 设置精度:"medium" 或者 "high"
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

# config
def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

config = load_config()

root = config['root']
EPOCHS = config['EPOCHS']
OUT_CLASSES = config['OUT_CLASSES']
idx_number = config['idx_number']
model_name = config['model_name'][1]                    # 1: U-Net++
encoder_model_name = config['encoder_model_name'][2]    # 2: ResNet-50
batch_size_num = config['batch_size']
n_cpu = os.cpu_count()

print(f"Selected model: {model_name}")
print(f"Selected encoder model: {encoder_model_name}")

class PetModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        # Calculate metrics
        acc = (tp + tn) / (tp + fp + tn + fn)  # Accuracy
        inacc = (fp + fn) / (tp + fp + tn + fn)  # Inaccuracy

        # Check if tp + fp is non-zero for precision and sensitivity calculation
        pre = (tp.sum() / (tp + fp).sum()) if (tp + fp).sum() != 0 else 0  # Precision
        sen = (tp.sum() / (tp + fn).sum()) if (tp + fn).sum() != 0 else 0  # Sensitivity

        return {
            "loss": loss,
            "acc": acc,
            "inacc": inacc,
            "pre": pre,
            "sen": sen,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Calculate IOU
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        # Calculate accuracy, inaccuracy, precision, and sensitivity
        acc = (tp + tn) / (tp + fp + tn + fn)  # Accuracy
        inacc = (fp + fn) / (tp + fp + tn + fn)  # Inaccuracy
        pre = torch.mean(torch.stack([x["pre"] for x in outputs]))  # Precision
        sen = torch.mean(torch.stack([x["sen"] for x in outputs]))  # Sensitivity

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": acc.mean(),
            f"{stage}_inaccuracy": inacc.mean(),
            f"{stage}_precision": pre,
            f"{stage}_sensitivity": sen,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        loss = train_loss_info["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        avg_train_loss = torch.mean(torch.stack([x["loss"] for x in self.training_step_outputs]))
        print(f"Epoch {self.current_epoch}, Average Training Loss: {avg_train_loss.item()}")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        loss = valid_loss_info["loss"]
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        loss = test_loss_info["loss"]
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader), eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def visualize_results(model, test_dataloader, logger, idx_number=5):
    print("Saving Visualize Results ... ...")
    log_dir = logger.log_dir
    save_dir = os.path.join(log_dir, "segmentation_results")
    os.makedirs(save_dir, exist_ok=True)

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
        if idx <= (idx_number-1):
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))

            axes[0].imshow(image.numpy().transpose(1, 2, 0))
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(gt_mask.numpy().squeeze())
            axes[1].set_title("Ground truth")
            axes[1].axis("off")

            axes[2].imshow(pr_mask.numpy().squeeze())
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            save_path = os.path.join(save_dir, f"segmentation_{idx}.png")
            plt.savefig(save_path)
            plt.close(fig)


def train_model(trainer, model, train_dataloader, valid_dataloader):
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def validate_model(trainer, model, valid_dataloader):
    return trainer.validate(model, dataloaders=valid_dataloader, verbose=False)


def test_model(trainer, model, test_dataloader):
    return trainer.test(model, dataloaders=test_dataloader, verbose=False)


if __name__ == "__main__":

    # Argument parser -------------------------------------------------
    parser = argparse.ArgumentParser(description="Image Segmentation with Seg-model")
    parser.add_argument('--train', action='store_true', help="Whether to train the model")
    parser.add_argument('--load_model', type=str, default=None, help="Path to load the pre-trained model")

    args = parser.parse_args()

    # Dataset -------------------------------------------------
    train_dataset = SimpleOxfordPetDataset(root, "train")
    valid_dataset = SimpleOxfordPetDataset(root, "valid")
    test_dataset = SimpleOxfordPetDataset(root, "test")

    # Dataset size check
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_num, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_num, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_num, shuffle=False, num_workers=n_cpu)

    model = PetModel(model_name, encoder_model_name, in_channels=3, out_classes=1)

    log_name = model_name + "_" + encoder_model_name + "_segmentation"
    logger = TensorBoardLogger("tb_logs", name=log_name)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, logger=logger)

    if args.train:
        train_model(trainer, model, train_dataloader, valid_dataloader)
    elif args.load_model:
        model = PetModel.load_from_checkpoint(args.load_model, arch=model_name, encoder_name=encoder_model_name, in_channels=3, out_classes=1)
        print(f"Model loaded from {args.load_model}")

    # Validate and test model -------------------------------------------------
    valid_metrics = validate_model(trainer, model, valid_dataloader)
    print(valid_metrics)

    test_metrics = test_model(trainer, model, test_dataloader)
    print(test_metrics)

    # Visualize results -------------------------------------------------
    visualize_results(model, test_dataloader, logger, idx_number)
    