import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # No gradient needed during evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)

            # Track metrics
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * running_correct / running_total

    return avg_loss, accuracy