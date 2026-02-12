import torch
from ignite.metrics import Precision,Recall,Accuracy,ConfusionMatrix,TopKCategoricalAccuracy
import torchvision.transforms as transforms
import torchvision

def evaluate(model, dataset_name, dataset_dir, device=None):
    """
    Evaluate model using ignite metrics.
    Returns: (accuracy, top5_acc, precision, recall, conf_matrix, sub_conf_matrix, acc_50)
    """
    # ---- Get dataloader (implement this) ----
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # test data
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    # test_loader = get_test_loader(dataset_name, dataset_dir)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ---- Ignite metrics ----
    accuracy_metric = Accuracy()
    top5_metric = TopKCategoricalAccuracy(k=5)
    precision_metric = Precision(average=True)      # macro by default
    recall_metric = Recall(average=True)            # macro by default
    # ConfusionMatrix returns a tensor; we will convert to list later
    conf_metric = ConfusionMatrix(num_classes=10)   # Adjust num_classes!
    # For binary threshold 0.5, we can either use Accuracy with threshold or compute manually
    # We'll compute acc_50 manually for clarity

    # ---- Accumulate over batches ----
    all_preds = []
    all_labels = []
    all_probs = []   # for acc_50 (binary only)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Update ignite metrics
            accuracy_metric.update((outputs, labels))
            top5_metric.update((outputs, labels))
            precision_metric.update((outputs, labels))
            recall_metric.update((outputs, labels))
            conf_metric.update((outputs, labels))

            # Also collect for acc_50 if binary
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    # ---- Compute final metric values ----
    acc = accuracy_metric.compute().item()
    tacc = top5_metric.compute().item()
    prec = precision_metric.compute().item()
    rec = recall_metric.compute().item()
    conf_matrix = conf_metric.compute().cpu().numpy().tolist()   # JSON serializable

    # ---- Sub confusion matrix (e.g., first 10 classes) ----
    num_classes = conf_metric.num_classes
    sub_classes = list(range(min(10, num_classes)))
    if conf_matrix:
        # extract submatrix
        sub_conf = [[conf_matrix[i][j] for j in sub_classes] for i in sub_classes]
    else:
        sub_conf = []

    # ---- Accuracy at threshold 0.5 (binary only) ----
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if num_classes == 2:
        pos_probs = all_probs[:, 1]
        preds_thresh = (pos_probs >= 0.5).int()
        acc_50 = (preds_thresh == all_labels).float().mean().item()
    else:
        acc_50 = acc   # or 0.0; adjust as needed

    # ---- Reset metrics for next call (important!) ----
    accuracy_metric.reset()
    top5_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    conf_metric.reset()

    return acc, tacc, prec, rec, conf_matrix, sub_conf, acc_50