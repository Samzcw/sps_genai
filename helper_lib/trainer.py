import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """
    Enhanced training loop with checkpoint functionality
    """

    datalogs = []
    best_accuracy = 0.0
    best_path = None

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.argmax(outputs.data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()  

            if (batch_number % 100 == 99):
                train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 
                                                        'avg loss': f'{running_loss/(batch_number+1):.4f}'})

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        # Evaluate on validation data after the epoch
        val_loss, val_accuracy = evaluate_model(
            model, val_loader, criterion, device
        )

        datalogs.append({
            "epoch": epoch + 1, 
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Save checkpoint every epoch using validation metrics
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, val_loss, val_accuracy, checkpoint_dir=checkpoint_dir
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_accuracy, 
                checkpoint_dir=f"{checkpoint_dir}/best"
            )
            print(f"New best model saved! Accuracy: {val_accuracy:.2f}%")
            print(f"Best model path: {best_path}")

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Accuracy={epoch_accuracy:.2f}%")
        print(f"Validation Loss={val_loss:.4f}, Validation Accuracy={val_accuracy:.2f}%")
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished Training")
    return model, best_path, datalogs