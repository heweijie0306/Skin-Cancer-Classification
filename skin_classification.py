import pandas as pd
import os
import numpy as np
import torch.optim
from sklearn.manifold import TSNE
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Subset
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from baseline import CNN
class ISICDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None, sample_ratio=1.0):
        self.images_path = images_path
        self.transform = transform
        self.df = pd.read_csv(csv_path, low_memory=False)
        allowed_classes = ['actinic keratosis', 'basal cell carcinoma', 'seborrheic keratosis', 'dermatofibroma', 'melanoma', 'vascular lesion']
        self.df = self.df[self.df['diagnosis'].isin(allowed_classes)]
        self.class_to_idx = self._get_class_to_idx()
    def _get_class_to_idx(self):
        """
        Returns a dictionary with class names as keys and their corresponding index as values.
        """
        class_to_idx = {class_name: i for i, class_name in enumerate(self.df['diagnosis'].unique())}
        return class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.images_path, self.df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)

        # Load label

        label = self.df.iloc[idx]['diagnosis']
        label = self.class_to_idx[label]
        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, label


def get_stats(preds, targets, num_classes):

    # Get conf matrix
    gt_label = np.arange(num_classes)
    conf_mat = confusion_matrix(targets, preds, labels=gt_label, normalize='true')
    class_correct = np.diag(conf_mat)


    return class_correct, conf_mat


def test(model, test_loader, device, criterion):

    model.eval()
    test_loss = 0
    correct = 0
    features = []
    with torch.no_grad():
        preds = []
        targets = []
        probs = []
        batch_features = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
            probs.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())
            batch_features.append(output.detach().cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    probs = np.concatenate(probs)
    features = np.concatenate(batch_features)
    return test_loss, test_acc, preds, targets, probs, features


def train(model, train_loader, optimizer, criterion, epochs,
          log_interval, device):
    model.train()
    per_epoch_loss = []
    per_epoch_acc = []
    features = []
    for epoch in range(epochs):
        train_loss = 0
        preds = []
        targets = []
        correct = 0
        batch_features = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Get the accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Save the predictions and targets if it's the last epoch
            if epoch == epochs - 1:
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                batch_features.append(output.detach().cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        per_epoch_acc.append(train_acc)
        if epoch % log_interval == 0:
            print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, train_loss, train_acc))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    features = np.concatenate(batch_features)
    return model, np.array(per_epoch_loss), np.array(per_epoch_acc), preds, targets, features


def visulization(conf_mats, num_folds, class_names, num_classes, matrix_type='train'):

    avg_conf_mat = np.zeros_like(conf_mats[0])

    for conf_mat in conf_mats:
        avg_conf_mat += conf_mat

    avg_conf_mat /= num_folds

    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(avg_conf_mat, display_labels=class_names)
    disp.plot(include_values=False, cmap=plt.cm.Reds, xticks_rotation='vertical', ax=ax)
    disp.ax_.get_images()[0].set_clim(0, 1)

    if matrix_type == 'train':
        ax.set_title(f'Overall training confusion matrix')
        fig.tight_layout()
        plt.savefig('overall_train_conf_mat.png')
        plt.close()

        overall_per_class_train = avg_conf_mat.diagonal()
        overall_train_std = np.std(overall_per_class_train)
        overall_train_mean = np.mean(overall_per_class_train)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(num_classes), overall_per_class_train)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Class')
        ax.set_title(f'Overall per class training accuracy')
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation='vertical')

        # Add two horizontal lines for overall std
        ax.axhline(y=overall_train_mean, color='r', linestyle='-', label='Mean')
        ax.axhline(y=overall_train_mean + overall_train_std, color='g', linestyle='--', label='Mean+-Std')
        ax.axhline(y=overall_train_mean - overall_train_std, color='g', linestyle='--')
        ax.legend()

        fig.tight_layout()
        plt.savefig('overall_per_class_train.png')
        plt.close()

    else:
        ax.set_title(f'Overall testing confusion matrix')
        fig.tight_layout()
        plt.savefig('overall_test_conf_mat.png')
        plt.close()
        overall_per_class_test = avg_conf_mat.diagonal()
        overall_test_std = np.std(overall_per_class_test)
        overall_test_mean = np.mean(overall_per_class_test)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(num_classes), overall_per_class_test)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Class')
        ax.set_title(f'Overall per class testing accuracy')
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation='vertical')
        # Add two horizontal lines for overall std
        ax.axhline(y=overall_test_mean, color='r', linestyle='-', label='Mean')
        ax.axhline(y=overall_test_mean + overall_test_std, color='g', linestyle='--', label='Mean+-Std')
        ax.axhline(y=overall_test_mean - overall_test_std, color='g', linestyle='--')
        ax.legend()
        fig.tight_layout()
        plt.savefig('overall_per_class_test.png')


def plot_tsne(features, labels, class_names, matrix_type='train'):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        plt.scatter(reduced_features[labels == i, 0], reduced_features[labels == i, 1], label=class_name)

    if matrix_type == 'train':
        plt.title('t-SNE Visualization of Train Features')
        plt.legend()
        plt.savefig('train_tsne_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.title('t-SNE Visualization of Test Features')
        plt.legend()
        plt.savefig('test_tsne_plot.png', dpi=300, bbox_inches='tight')
        plt.close()


def ROC(num_classes, targets, probs):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot ROC curves
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('ROC Curve')
    plt.close()



def plot_sensitivity_specificity_auc(conf_mats, num_folds, targets, probs, class_names):
    num_classes = len(class_names)
    # Calculate confusion matrix
    avg_conf_mat = np.zeros_like(conf_mats[0])

    for conf_mat in conf_mats:
        avg_conf_mat += conf_mat
    avg_conf_mat /= num_folds

    true_pos = np.diag(avg_conf_mat)
    false_pos = np.sum(avg_conf_mat, axis=0) - true_pos
    false_neg = np.sum(avg_conf_mat, axis=1) - true_pos
    true_neg = np.sum(avg_conf_mat) - (true_pos + false_pos + false_neg)

    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve, sensitivity, and specificity
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
        plt.scatter(1 - specificity[i], sensitivity[i], label=f'Sensitivity/Specificity of class {i}', marker='o')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic (ROC) and Sensitivity/Specificity')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Sensitivity_Specificity.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    log_interval = 2
    n_splits = 5  # Number of folds for cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    csv_path = 'C:/Users/Ethan He/images/metadata.csv'
    images_path = "C:/Users/Ethan He/images"

    dataset = ISICDataset(csv_path, images_path, transform=transform, sample_ratio=0.1)
    dataset_indices = list(range(len(dataset)))

    num_classes = len(set(dataset.df['diagnosis'].unique()))
    print("number_classes", num_classes)
    class_names = list(dataset.class_to_idx.keys())

    batch_size = 64

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    # # resnet.load_state_dict(torch.load('trained_resnet50_v221.pth'))
    # resnet = CNN(input_channels=3, img_size=224, num_classes=num_classes).to(device)
    resnet = resnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)



    test_accuracies = []
    conf_train_mats = []
    conf_test_mats = []
    fold = 1

    for train_data_indices, test_data_indices in kfold.split(dataset_indices):
        print(f"Fold {fold}/{n_splits}")
        train_data = Subset(dataset, train_data_indices)
        test_data = Subset(dataset, test_data_indices)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        model, per_epoch_loss, per_epoch_acc, train_preds, train_targets, train_features = train(resnet, train_loader,
                                                                                 optimizer, criterion,
                                                                                 num_epochs,
                                                                                 log_interval, device)

        test_loss, test_acc, test_preds, test_targets, test_probs, test_features = test(resnet, test_loader, device, criterion)
        test_accuracies.append(test_acc)
        classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)
        classes_train, overall_train_mat= get_stats(train_preds, train_targets, num_classes)
        conf_train_mats.append(overall_train_mat)
        conf_test_mats.append(overall_test_mat)
        print(f'Test accuracy for fold {fold}: {test_acc * 100:.3f}')
        fold += 1
        if fold == n_splits:
            plot_tsne(train_features, train_targets, class_names, matrix_type='train')
            plot_tsne(test_features, test_targets, class_names, matrix_type='test')

    print(f'Average Test accuracy: {np.mean(test_accuracies) * 100:.3f}')
    torch.save(model.state_dict(), "trained_CNN.pth")
    visulization(conf_train_mats, n_splits, class_names, num_classes, matrix_type='train')
    visulization(conf_test_mats, n_splits, class_names, num_classes, matrix_type='test')
    ROC(num_classes, test_targets, test_probs)
    plot_sensitivity_specificity_auc(conf_test_mats, n_splits, test_targets, test_probs, class_names)




if __name__ == '__main__':
    main()

