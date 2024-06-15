"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import logging
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from tqdm.auto import tqdm

from certify import evaluate
from utils import remove_oldest_files, load_components, setup_tqdm


def train_batch(batch, model, loss, optimizer, scheduler, grad_scaler, device):
    inputs, attention_masks, labels = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )
    inputs, attention_masks, labels = (
        inputs.to(device),
        attention_masks.to(device),
        labels.to(device),
    )

    optimizer.zero_grad()
    logits = model(inputs, attention_mask=attention_masks).logits
    preds = logits.argmax(dim=1)

    loss_value = loss(logits, labels)
    loss_value.backward()
    
    if grad_scaler:
        grad_scaler(model=model)
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss_value.item(), preds.detach(), labels


def train_epoch(epoch, model, dataloader, loss, optimizer, scheduler, grad_scaler, device, update_step=10):
    tqdm_params = setup_tqdm(total=len(dataloader), desc=f"Epoch: {epoch}")
    with tqdm(**tqdm_params) as progress_bar:
        steps_since_update = 0
        total_loss, total_corrects, total_samples = 0, 0, 0

        for batch in dataloader:
            loss_value, preds, labels = train_batch(
                batch=batch, model=model, loss=loss, optimizer=optimizer, scheduler=scheduler, grad_scaler=grad_scaler, device=device,
            )
            total_loss += loss_value
            total_corrects += (preds == labels).sum().item()
            total_samples += len(labels)

            avg_loss = total_loss / (total_samples / len(labels))
            avg_accuracy = total_corrects / total_samples
            steps_since_update += 1
            if steps_since_update >= update_step:
                progress_bar.update(steps_since_update)
                progress_bar.set_postfix(
                    {
                        "Batch Loss": f"{loss_value:.4f}",
                        "Epoch Avg. Loss": f"{avg_loss:.4f}",
                        "Epoch Avg. Acc.": f"{avg_accuracy:.2%}",
                    },
                    refresh=True,
                )

                message = (
                    f"Epoch: {epoch:<5} | "
                    f"Batch Loss: {loss_value:<10.4f} | "
                    f"Epoch Avg. Loss: {avg_loss:<10.4f} | "
                    f"Epoch Avg. Accuracy: {avg_accuracy:<7.2%}"
                )
                logging.debug(message)
                steps_since_update = 0

        # Update for any remaining steps
        if steps_since_update:
            progress_bar.update(steps_since_update)

    return total_loss / len(dataloader), total_corrects / total_samples


def train_model(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    components = load_components(config)

    model = components["model"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]
    grad_scaler = components["grad_scaler"]
    loss_function = components["loss_function"]
    train_loader = components["train_loader"]
    valid_dataset = components["valid_dataset"]
    num_labels = components["num_labels"]
    device = components["device"]

    checkpoint_dir = config["checkpoint_dir"]
    pred_dir = config["pred_dir"]
    log_dir = config["log_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    evaluate_epoch = config["evaluate_epoch"]
    num_samples = config["num_samples"]
    history = defaultdict(dict)
    patience = config["early_stopping_patience"]
    best_val_acc = 0
    epochs_without_improvement = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        tr_loss, tr_acc = train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            loss=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            device=device,
            update_step=config["update_step"],
        )

        message = (
            f"Epoch: {epoch:<5} | "
            f"Epoch Train Loss: {tr_loss:<10.4f} | "
            f"Epoch Train Accuracy: {tr_acc:<7.2%}"
        )
        logging.info(message)

        # Store training metrics in history
        history[epoch]["tr_loss"] = tr_loss
        history[epoch]["tr_acc"] = tr_acc

        # Evaluate and save checkpoints every evaluate_epoch
        if (epoch + 1) % evaluate_epoch == 0:
            val_acc, val_df = evaluate(
                model=model,
                dataset=valid_dataset,
                pred_num_samples=num_samples,
                batch_size=config["batch_size"],
                device=device,
            )
            history[epoch]["val_acc"] = val_acc
            message = (
                f"Epoch: {epoch:<5} | "
                f"Valid Accuracy (Best): {val_acc:<7.2%} ({best_val_acc:<7.2%})"
            )
            tqdm.write(message)
            logging.info(message)

            # Save model checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"epoch_{epoch}_checkpoint.pth"
            )
            checkpoint = {
                "epoch": epoch,
                "num_labels": num_labels,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None
                if scheduler is None else scheduler.state_dict(),
                "tr_loss": tr_loss,
                "tr_acc": tr_acc,
                "val_acc": val_acc,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            torch.save(checkpoint, checkpoint_path)
            remove_oldest_files(
                checkpoint_dir,
                keep=config["keep_checkpoints"],
                exclude_files={"best_checkpoint.pth"},
            )

            # Save the prediction DataFrame for the epoch
            preds_path = os.path.join(pred_dir, f"epoch_{epoch}_predictions.csv")
            val_df.to_csv(preds_path, index=False)

            # Early stopping check and best checkpoint saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = os.path.join(
                    checkpoint_dir, "best_checkpoint.pth"
                )
                torch.save(
                    checkpoint,
                    best_checkpoint_path,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logging.info(
                        f"Early stop at epoch {epoch}. No gain in {patience} evaluations."
                    )
                    break
    logging.info("Training completed.")
    df = pd.DataFrame.from_dict(history, orient="index").reset_index().rename(columns={'index': 'epoch'})
    history_csv_path = os.path.join(log_dir, "history.csv")
    df.to_csv(history_csv_path)
    logging.info(f"Training history saved to {history_csv_path}")
    return history
