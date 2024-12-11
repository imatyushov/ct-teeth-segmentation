# from with_nnunet import preprocess
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from random import random
import matplotlib

prediction_path = '../Multi_Oral_Structure/result/ToothFairy2F_007.mha'
original_image_path = '../Multi_Oral_Structure/nnUNet_raw/Dataset112_ToothFairy2/imagesTr/ToothFairy2F_007_0000.mha'
truth_mask_path = '../Multi_Oral_Structure/nnUNet_raw/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_007.mha'
colors = [(1, 1, 1)] + [(random(), random(), random()) for i in range(255)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=48)


def load_mha(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return array

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    if np.sum(y_true) + np.sum(y_pred) == 0:
        return 1  # Perfect match scenario
    return 2. * intersection / (np.sum(y_true) + np.sum(y_pred))

def calculate_dice_scores(gt_mask, pred_mask):
    labels = np.unique(np.concatenate([np.unique(gt_mask), np.unique(pred_mask)]))  # All unique labels from both masks
    dice_scores = {}

    for label in labels:
        gt_label_mask = (gt_mask == label)
        pred_label_mask = (pred_mask == label)
        dice_score = dice_coefficient(gt_label_mask, pred_label_mask)
        dice_scores[label] = dice_score

    return dice_scores

if __name__ == '__main__':

    t_mask = load_mha(truth_mask_path)
    pred_mask = load_mha(prediction_path)
    orig_image = load_mha(original_image_path)

    dice_scores = calculate_dice_scores(t_mask, pred_mask)
    for label, score in dice_scores.items():
        print(f"Dice score for label {label}: {score}")

    print(pred_mask.shape[0])

    # Select a slice to visualize if it's a 3D volume
    slice_index = pred_mask.shape[0] // 2  # Example: mid slice of a 3D volume

    # Display the original and the prediction
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image[slice_index], cmap='gray')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(orig_image[slice_index], cmap='gray')
    plt.imshow(pred_mask[slice_index], alpha=0.5, cmap=new_map)  # Red overlay for the segmentation
    plt.title('Prediction Overlay')
    plt.colorbar()
    plt.show()




    # data_dir = 'F:/Work/nnUnet_v2/dataset/nnUnet_raw/Dataset112_ToothFairy2'
    # preprocess.prepare_dataset(data_dir)


    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    # transform_mask = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])
    #
    # dataset = ImageDataset(data_dir, label_dir, transform, transform_mask)
    #
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    #
    # dataloader_train = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # dataloader_test = DataLoader(test_dataset, batch_size=4, shuffle=False)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # model = SimpleSegmentationModel(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # model.to(device)
    #
    # model.fit(dataloader_train, dataloader_train, optimizer=optimizer, epochs=10)


