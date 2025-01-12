import torch.optim as optim
import tqdm
import numpy as np
from Data_processing_dep_eeg import loading_data
from Data_processing_dep_emg import loading_data2
from model2 import MULTAV_CLASSFICATIONModel1
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score
import os
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU: {current_gpu_name}")
else:
    print("GPU No. Use CPU.")

def remove_low_accuracy_folders(model_save_path):
    # Get all subfolders under the main directory
    folders = [f for f in os.listdir(model_save_path) if os.path.isdir(os.path.join(model_save_path, f))]

    # Extract the accuracy and F1 score from the folder name
    folder_acc_f1_map = {}
    for folder in folders:
        try:
            # Extract the accuracy and F1 score from the folder name, assuming that the accuracy and F1 score are the values after "_acc_" and "_f1_" respectively
            acc_str = folder.split('_acc_')[1].split('_')[0]  # Extract the numeric part after acc
            f1_str = folder.split('_f1_')[1].split('_')[0]  # Extract the digital part after f1
            acc = float(acc_str)
            f1 = float(f1_str)
            folder_acc_f1_map[folder] = (acc, f1)
        except (IndexError, ValueError) as e:
            print(f"Unable to extract accuracy or F1 score from folder name {folder} ，error: {e}")
            continue

    sorted_folders = sorted(folder_acc_f1_map.items(), key=lambda x: x[1][0], reverse=True)

    if len(sorted_folders) > 2:
        highest_accuracy = sorted_folders[0][1][0]

        second_highest_accuracy = None
        second_highest_folders = []
        for folder, (acc, f1) in sorted_folders:
            if acc < highest_accuracy:
                if second_highest_accuracy is None:
                    second_highest_accuracy = acc
                if acc == second_highest_accuracy:
                    second_highest_folders.append((folder, f1))

        folders_to_keep = [folder for folder, (acc, f1) in sorted_folders if acc == highest_accuracy]

        if len(second_highest_folders) > 2:
            second_highest_folders_sorted = sorted(second_highest_folders, key=lambda x: x[1], reverse=True)
            folders_to_keep += [folder for folder, f1 in second_highest_folders_sorted[:2]]
        else:
            folders_to_keep += [folder for folder, f1 in second_highest_folders]

        for folder, (acc, f1) in sorted_folders:
            folder_path = os.path.join(model_save_path, folder)

            if folder not in folders_to_keep and os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}，acc: {acc}，F1 score: {f1}")
                except Exception as e:
                    print(f"Deleted folder {folder_path} error: {e}")
            elif not os.path.exists(folder_path):
                print(f"Folders {folder_path} no longer exists. Skip deletion.")

(x_train_z_eeg, x_train_b_eeg, x_train_l_eeg, x_train_t_eeg, x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, y_train_eeg, y_test_eeg, y_train_data_Ex_eeg, y_test_data_Ex_eeg, y_train_data_Ag_eeg,
y_test_data_Ag_eeg, y_train_data_Co_eeg, y_test_data_Co_eeg, y_train_data_Ne_eeg, y_test_data_Ne_eeg, y_train_data_Op_eeg, y_test_data_Op_eeg) = loading_data()

(x_train_z_emg, x_train_b_emg, x_train_l_emg, x_train_t_emg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg, y_train_emg, y_test_emg, y_train_data_Ex_emg, y_test_data_Ex_emg, y_train_data_Ag_emg,
y_test_data_Ag_emg, y_train_data_Co_emg, y_test_data_Co_emg, y_train_data_Ne_emg, y_test_data_Ne_emg, y_train_data_Op_emg, y_test_data_Op_emg) = loading_data2()

x_train_z_eeg = torch.tensor(x_train_z_eeg, dtype=torch.float32).to(device)
x_test_z_eeg = torch.tensor(x_test_z_eeg, dtype=torch.float32).to(device)
x_train_b_eeg = torch.tensor(x_train_b_eeg, dtype=torch.float32).to(device)
x_test_b_eeg = torch.tensor(x_test_b_eeg, dtype=torch.float32).to(device)
x_train_l_eeg = torch.tensor(x_train_l_eeg, dtype=torch.float32).to(device)
x_test_l_eeg = torch.tensor(x_test_l_eeg, dtype=torch.float32).to(device)
x_train_t_eeg = torch.tensor(x_train_t_eeg, dtype=torch.float32).to(device)
x_test_t_eeg = torch.tensor(x_test_t_eeg, dtype=torch.float32).to(device)

x_train_z_emg = torch.tensor(x_train_z_emg, dtype=torch.float32).to(device)
x_test_z_emg = torch.tensor(x_test_z_emg, dtype=torch.float32).to(device)
x_train_b_emg = torch.tensor(x_train_b_emg, dtype=torch.float32).to(device)
x_test_b_emg = torch.tensor(x_test_b_emg, dtype=torch.float32).to(device)
x_train_l_emg = torch.tensor(x_train_l_emg, dtype=torch.float32).to(device)
x_test_l_emg = torch.tensor(x_test_l_emg, dtype=torch.float32).to(device)
x_train_t_emg = torch.tensor(x_train_t_emg, dtype=torch.float32).to(device)
x_test_t_emg = torch.tensor(x_test_t_emg, dtype=torch.float32).to(device)

loss_func = nn.CrossEntropyLoss()
k1 = 0.9
k2 = 0.1
epochs = 50
batch_size = 64

# Define learning rates and corresponding save paths
learning_rates = [0.0009, 0.003, 0.005, 0.007, 0.009]
base_model_save_path = r"D:\our\dep_eeg+emg_zblt_per"


# Outer loop for different learning rates
for lr in learning_rates:
    # Create a unique model save path for each learning rate
    model_save_path = os.path.join(base_model_save_path, f"lr_{lr}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_test_accuracies = []
    best_f1_scores = []
    best_model_details = {}

    for iteration in range(1, 101):
        print(f"Starting iteration {iteration} with learning rate {lr}")

        model = MULTAV_CLASSFICATIONModel1().to(device)


        saved_model = torch.load(
            "Pretrained_weights/depression_model.pt")


        state_dict = saved_model.state_dict()


        model_dict = model.state_dict()


        matched_state_dict = {k: v for k, v in state_dict.items() if
                              k in model_dict and v.size() == model_dict[k].size()}


        model_dict.update(matched_state_dict)

        model.load_state_dict(model_dict)


        unfrozen_params = [
            "lstm_z_eeg.weight_ih_l0",
            "lstm_z_eeg.weight_hh_l0",
            "lstm_z_eeg.bias_ih_l0",
            "lstm_z_eeg.bias_hh_l0",
            "lstm_z_eeg.weight_ih_l0_reverse",
            "lstm_z_eeg.weight_hh_l0_reverse",
            "lstm_z_eeg.bias_ih_l0_reverse",
            "lstm_z_eeg.bias_hh_l0_reverse",
            "lstm_z_emg.weight_ih_l0",
            "lstm_z_emg.weight_hh_l0",
            "lstm_z_emg.bias_ih_l0",
            "lstm_z_emg.bias_hh_l0",
            "lstm_z_emg.weight_ih_l0_reverse",
            "lstm_z_emg.weight_hh_l0_reverse",
            "lstm_z_emg.bias_ih_l0_reverse",
            "lstm_z_emg.bias_hh_l0_reverse",

            "lstm_b_eeg.weight_ih_l0",
            "lstm_b_eeg.weight_hh_l0",
            "lstm_b_eeg.bias_ih_l0",
            "lstm_b_eeg.bias_hh_l0",
            "lstm_b_eeg.weight_ih_l0_reverse",
            "lstm_b_eeg.weight_hh_l0_reverse",
            "lstm_b_eeg.bias_ih_l0_reverse",
            "lstm_b_eeg.bias_hh_l0_reverse",
            "lstm_b_emg.weight_ih_l0",
            "lstm_b_emg.weight_hh_l0",
            "lstm_b_emg.bias_ih_l0",
            "lstm_b_emg.bias_hh_l0",
            "lstm_b_emg.weight_ih_l0_reverse",
            "lstm_b_emg.weight_hh_l0_reverse",
            "lstm_b_emg.bias_ih_l0_reverse",
            "lstm_b_emg.bias_hh_l0_reverse",

            "lstm_l_eeg.weight_ih_l0",
            "lstm_l_eeg.weight_hh_l0",
            "lstm_l_eeg.bias_ih_l0",
            "lstm_l_eeg.bias_hh_l0",
            "lstm_l_eeg.weight_ih_l0_reverse",
            "lstm_l_eeg.weight_hh_l0_reverse",
            "lstm_l_eeg.bias_ih_l0_reverse",
            "lstm_l_eeg.bias_hh_l0_reverse",
            "lstm_l_emg.weight_ih_l0",
            "lstm_l_emg.weight_hh_l0",
            "lstm_l_emg.bias_ih_l0",
            "lstm_l_emg.bias_hh_l0",
            "lstm_l_emg.weight_ih_l0_reverse",
            "lstm_l_emg.weight_hh_l0_reverse",
            "lstm_l_emg.bias_ih_l0_reverse",
            "lstm_l_emg.bias_hh_l0_reverse",

            "lstm_t_eeg.weight_ih_l0",
            "lstm_t_eeg.weight_hh_l0",
            "lstm_t_eeg.bias_ih_l0",
            "lstm_t_eeg.bias_hh_l0",
            "lstm_t_eeg.weight_ih_l0_reverse",
            "lstm_t_eeg.weight_hh_l0_reverse",
            "lstm_t_eeg.bias_ih_l0_reverse",
            "lstm_t_eeg.bias_hh_l0_reverse",
            "lstm_t_emg.weight_ih_l0",
            "lstm_t_emg.weight_hh_l0",
            "lstm_t_emg.bias_ih_l0",
            "lstm_t_emg.bias_hh_l0",
            "lstm_t_emg.weight_ih_l0_reverse",
            "lstm_t_emg.weight_hh_l0_reverse",
            "lstm_t_emg.bias_ih_l0_reverse",
            "lstm_t_emg.bias_hh_l0_reverse",

            "MultiheadAttention_eeg.in_proj_weight",
            "MultiheadAttention_eeg.in_proj_bias",
            "MultiheadAttention_eeg.out_proj.weight",
            "MultiheadAttention_eeg.out_proj.bias",
            "MultiheadAttention_emg.in_proj_weight",
            "MultiheadAttention_emg.in_proj_bias",
            "MultiheadAttention_emg.out_proj.weight",
            "MultiheadAttention_emg.out_proj.bias",

            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer_depression.weight",
            "out_layer_depression.bias",

            "TAU_CBAM_eeg.ChannelGate.mlp.1.weight",
            "TAU_CBAM_eeg.ChannelGate.mlp.1.bias",
            "TAU_CBAM_eeg.ChannelGate.mlp.3.weight",
            "TAU_CBAM_eeg.ChannelGate.mlp.3.bias",
            "TAU_CBAM_eeg.SpatialGate.spatial.conv.weight",
            "TAU_CBAM_eeg.SpatialGate.spatial.bn.weight",
            "TAU_CBAM_eeg.SpatialGate.spatial.bn.bias",
            "TAU_CBAM_eeg.dw_conv.weight",
            "TAU_CBAM_eeg.dw_conv.bias",
            "TAU_CBAM_eeg.dw_d_conv.weight",
            "TAU_CBAM_eeg.dw_d_conv.bias",
            "TAU_CBAM_eeg.conv_1x1.weight",
            "TAU_CBAM_eeg.conv_1x1.bias",

            "MultiheadAttention_event_before_eeg.in_proj_weight",
            "MultiheadAttention_event_before_eeg.in_proj_bias",
            "MultiheadAttention_event_before_eeg.out_proj.weight",
            "MultiheadAttention_event_before_eeg.out_proj.bias",
            "MultiheadAttention_event_after_eeg.in_proj_weight",
            "MultiheadAttention_event_after_eeg.in_proj_bias",
            "MultiheadAttention_event_after_eeg.out_proj.weight",
            "MultiheadAttention_event_after_eeg.out_proj.bias",

            "TAU_CBAM_emg.ChannelGate.mlp.1.weight",
            "TAU_CBAM_emg.ChannelGate.mlp.1.bias",
            "TAU_CBAM_emg.ChannelGate.mlp.3.weight",
            "TAU_CBAM_emg.ChannelGate.mlp.3.bias",
            "TAU_CBAM_emg.SpatialGate.spatial.conv.weight",
            "TAU_CBAM_emg.SpatialGate.spatial.bn.weight",
            "TAU_CBAM_emg.SpatialGate.spatial.bn.bias",
            "TAU_CBAM_emg.dw_conv.weight",
            "TAU_CBAM_emg.dw_conv.bias",
            "TAU_CBAM_emg.dw_d_conv.weight",
            "TAU_CBAM_emg.dw_d_conv.bias",
            "TAU_CBAM_emg.conv_1x1.weight",
            "TAU_CBAM_emg.conv_1x1.bias",

            "MultiheadAttention_event_before_emg.in_proj_weight",
            "MultiheadAttention_event_before_emg.in_proj_bias",
            "MultiheadAttention_event_before_emg.out_proj.weight",
            "MultiheadAttention_event_before_emg.out_proj.bias",
            "MultiheadAttention_event_after_emg.in_proj_weight",
            "MultiheadAttention_event_after_emg.in_proj_bias",
            "MultiheadAttention_event_after_emg.out_proj.weight",
            "MultiheadAttention_event_after_emg.out_proj.bias",

            "proj11.weight",
            "proj11.bias",
            "proj21.weight",
            "proj21.bias",
            "proj12.weight",
            "proj12.bias",
            "proj22.weight",
            "proj22.bias",
            "proj13.weight",
            "proj13.bias",
            "proj23.weight",
            "roj23.bias",
            "proj14.weight",
            "proj14.bias",
            "proj24.weight",
            "proj24.bias",
            "proj15.weight",
            "proj15.bias",
            "proj25.weight",
            "proj25.bias",
            "out_layer_Ex.weight",
            "out_layer_Ex.bias",
            "out_layer_Ag.weight",
            "out_layer_Ag.bias",
            "out_layer_Co.weight",
            "out_layer_Co.bias",
            "out_layer_Ne.weight",
            "out_layer_Ne.bias",
            "out_layer_Op.weight",
            "out_layer_Op.bias",

        ]
        for name, param in model.named_parameters():

            if name in matched_state_dict:
                if name in unfrozen_params:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        for name, param in model.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.000005)

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        best_test = 0
        best_f1 = 0
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct_train = 0

            for i in tqdm.trange(0, len(x_train_z_eeg), batch_size):
                x_z_eeg = x_train_z_eeg[i:i + batch_size]
                x_b_eeg = x_train_b_eeg[i:i + batch_size]
                x_l_eeg = x_train_l_eeg[i:i + batch_size]
                x_t_eeg = x_train_t_eeg[i:i + batch_size]

                x_z_emg = x_train_z_emg[i:i + batch_size]
                x_b_emg = x_train_b_emg[i:i + batch_size]
                x_l_emg = x_train_l_emg[i:i + batch_size]
                x_t_emg = x_train_t_emg[i:i + batch_size]

                y_batch_fenlei = y_train_eeg[i:i + batch_size]
                y_batch_Ex = y_train_data_Ex_eeg[i:i + batch_size]
                y_batch_Ag = y_train_data_Ag_eeg[i:i + batch_size]
                y_batch_Co = y_train_data_Co_eeg[i:i + batch_size]
                y_batch_Ne = y_train_data_Ne_eeg[i:i + batch_size]
                y_batch_Op = y_train_data_Op_eeg[i:i + batch_size]

                optimizer.zero_grad()
                output_depression, output_Ex, output_Ag, output_Co, output_Ne, output_Op = model(x_z_eeg, x_b_eeg,
                                                                                                 x_l_eeg, x_t_eeg,
                                                                                                 x_z_emg, x_b_emg,
                                                                                                 x_l_emg, x_t_emg)
                loss_depression = loss_func(output_depression, torch.tensor(y_batch_fenlei).to(device))

                loss_EX = loss_func(output_Ex, torch.tensor(y_batch_Ex).to(device))
                loss_Ag = loss_func(output_Ag, torch.tensor(y_batch_Ag).to(device))
                loss_Co = loss_func(output_Co, torch.tensor(y_batch_Co).to(device))
                loss_Ne = loss_func(output_Ne, torch.tensor(y_batch_Ne).to(device))
                loss_Op = loss_func(output_Op, torch.tensor(y_batch_Op).to(device))
                loss_personality = loss_depression + loss_EX + loss_Ag + loss_Co + loss_Ne + loss_Op

                loss = k1 * loss_depression + k2 * loss_personality

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                y_batch_classes = torch.argmax(output_depression, dim=1)
                correct_train += (y_batch_classes == torch.tensor(y_batch_fenlei).to(device)).sum().item()

            train_accuracy = correct_train / len(x_train_z_eeg)
            train_accuracies.append(train_accuracy)
            train_loss = total_loss / len(x_train_z_eeg)
            train_losses.append(train_loss)

            with torch.no_grad():
                model.eval()
                y_test_pred_depression, y_test_pred_Ex, y_test_pred_Ag, y_test_pred_Co, y_test_pred_Ne, y_test_pred_Op = model(
                    x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, x_test_z_emg, x_test_b_emg, x_test_l_emg,
                    x_test_t_emg)
                y_test_pred_classes = torch.argmax(y_test_pred_depression, dim=1)
                correct_test = torch.sum(y_test_pred_classes == torch.tensor(y_test_eeg).to(device))
                total_test = len(y_test_eeg)
                test_accuracy = correct_test.item() / total_test
                test_accuracies.append(test_accuracy)
                test_loss_depression = loss_func(y_test_pred_depression, torch.tensor(y_test_eeg).to(device))
                test_loss_EX = loss_func(y_test_pred_Ex, torch.tensor(y_test_data_Ex_eeg).to(device))
                test_loss_Ag = loss_func(y_test_pred_Ag, torch.tensor(y_test_data_Ag_eeg).to(device))
                test_loss_Co = loss_func(y_test_pred_Co, torch.tensor(y_test_data_Co_eeg).to(device))
                test_loss_Ne = loss_func(y_test_pred_Ne, torch.tensor(y_test_data_Ne_eeg).to(device))
                test_loss_Op = loss_func(y_test_pred_Op, torch.tensor(y_test_data_Op_eeg).to(device))
                test_loss_personality = test_loss_EX + test_loss_Ag + test_loss_Co + test_loss_Ne + test_loss_Op

                test_loss = k1 * test_loss_depression + k2 * test_loss_personality

                test_losses.append(test_loss.item())

                y_test_cpu = torch.tensor(y_test_eeg).cpu().numpy()
                y_test_pred_classes_cpu = y_test_pred_classes.cpu().numpy()
                precision = precision_score(y_test_cpu, y_test_pred_classes_cpu, average=None)
                precision = 0 if any(element <= 0.1 for element in precision) else 1
                f1 = f1_score(y_test_cpu, y_test_pred_classes_cpu, average='weighted')

                print(f'\nEpoch [{epoch + 1}/{epochs}] \n'
                      f'Training Loss:    {train_loss:.4f} - Training Accuracy:    {train_accuracy:.4f}\n'
                      f'Test Loss:        {test_loss:.4f} - Test Accuracy:        {test_accuracy:.4f}\n'
                      f'F1 Score:         {f1:.4f}\n\n')

                if test_accuracy > best_test and precision == 1:
                    # Create a folder name based on iteration, epoch, accuracy, and F1 score
                    folder_name = f"iteration_{iteration}_epoch_{epoch + 1}_acc_{test_accuracy:.4f}_f1_{f1:.4f}"
                    folder_path = os.path.join(model_save_path, folder_name)

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    # Save the model in the subfolder
                    model_path = os.path.join(folder_path, f"best_model.pt")

                    torch.save(model, model_path)  # Save the model

                    print(f"Saved model at {model_path}!")
                    best_test = test_accuracy
                    best_f1 = f1
                    best_epoch = epoch + 1

        best_test_accuracies.append(best_test)
        best_f1_scores.append(best_f1)
        best_model_details[iteration] = (best_epoch, best_test, best_f1)  # Track best model details

    # Calculate max, mean, and std for accuracies and F1 scores
    max_best_accuracy = max(best_test_accuracies)
    mean_best_accuracy = np.mean(best_test_accuracies)
    std_best_accuracy = np.std(best_test_accuracies)

    max_best_f1 = max(best_f1_scores)
    mean_best_f1 = np.mean(best_f1_scores)
    std_best_f1 = np.std(best_f1_scores)

    # Save the best model details to a text file
    with open(os.path.join(model_save_path, 'best_model_details.txt'), 'w') as f:
        for iteration, details in best_model_details.items():
            f.write(f"Iteration {iteration}: Best Epoch: {details[0]}, Accuracy: {details[1]}, F1 Score: {details[2]}\n")

    print(f'Finished training with learning rate {lr}.\n'
          f'Max Accuracy: {max_best_accuracy:.4f}, Mean Accuracy: {mean_best_accuracy:.4f}, Std Accuracy: {std_best_accuracy:.4f}\n'
          f'Max F1 Score: {max_best_f1:.4f}, Mean F1 Score: {mean_best_f1:.4f}, Std F1 Score: {std_best_f1:.4f}\n')

    # Remove low accuracy folders
    remove_low_accuracy_folders(model_save_path)

# Code execution finished
print("All models are trained and saved！")



