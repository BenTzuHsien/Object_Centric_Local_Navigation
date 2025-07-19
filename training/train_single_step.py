import os, torch, numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from Object_Centric_Local_Navigation.training.utils import get_least_used_gpu, get_top_available_gpus, plot_graph
from Object_Centric_Local_Navigation.training.single_step_dataset import SingleStepDataset

def train_single_step(model, dataset_path, result_path, 
                      num_gpus=1, transform=None, use_embedding=False, start_index=1):
    PARAM = {
        'Batch_Size': 128,
        'Learning_Rate': 1e-4,
        'Num_Epochs': 1000,
        'Weight_Saving_Step': 50
    }

    # Tracking Parameters
    training_losses = []
    accuracies = []

    # Setup Saving Path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    weight_save_dir = os.path.join(result_path, 'weights')
    if not os.path.exists(weight_save_dir):
        os.mkdir(weight_save_dir)
    training_losses_path = os.path.join(result_path, 'training_losses.npy')
    accuracies_path = os.path.join(result_path, 'accuracies.npy')

    # Setup dataset
    train_dataset = SingleStepDataset(dataset_path, transform, use_embeddings=use_embedding)
    train_dataloader = DataLoader(train_dataset, batch_size=PARAM['Batch_Size'], shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)
    goal_images, prompt = train_dataset.get_goal()

    # Setup model and device
    if num_gpus > 1:
        print('Multi GPU version still has some problem!!')
        pass
        # top_gpus = get_top_available_gpus(num_gpus)
        # primary_device = f'cuda:{top_gpus[0]}'
        # model = model.to(primary_device)
        # model.set_goal(goal_images, prompt)
        # model = torch.nn.DataParallel(model, device_ids=top_gpus)
        # DEVICE = primary_device
    else:
        least_used_gpu = get_least_used_gpu()
        DEVICE = f'cuda:{least_used_gpu}'
        model = model.to(DEVICE)
        model.set_goal(goal_images, prompt)
    model.train()

    # Resume previous training
    if start_index > 1:
        weight_path = os.path.join(result_path, 'weights', f'{start_index}.pth')
        model.load_weight(weight_path)
        print('Weight Loaded!')

        training_losses = list(numpy.load(training_losses_path))[:start_index-1]
        training_losses_path = os.path.join(result_path, 'new_training_losses.npy')
        accuracies = list(numpy.load(accuracies_path))[:int(start_index/PARAM['Weight_Saving_Step'] - 1)]
        accuracies_path = os.path.join(result_path, 'new_accuracies.npy')
        print('Tracking Parameter Loaded!')
    start_index -= 1

    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM['Learning_Rate'])

    # Training
    training_bar = tqdm(range(start_index, PARAM['Num_Epochs']), desc=f'Training {model.__class__.__name__}, Epochs')
    for epoch in training_bar:

        running_loss = 0.0
        for current_images, action in tqdm(train_dataloader, desc="Training", leave=False):

            action = action.to(DEVICE)
            optimizer.zero_grad()

            output, _ = model(current_images)
            output = output.permute(0, 2, 1)   # To accomadate how CrossEnropyLoss function accept as input (Batch_size, Num_classes, ...)
            loss = loss_fn(output, action)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_loss = running_loss / len(train_dataloader)
        training_losses.append([epoch+1, training_loss])
        training_bar.set_description(f'Training {model.__class__.__name__}, loss={training_loss:.4f}, Epochs')

        # Checkpoint
        if ((epoch + 1) % PARAM['Weight_Saving_Step']) == 0:
            weight_save_path = os.path.join(weight_save_dir, f'{epoch+1}.pth')
            torch.save(model.state_dict(), weight_save_path)
            tqdm.write(f'Save Weight {epoch+1}!')

            model.eval()
            with torch.no_grad():
                num_correct, num_total = 0, 0
                for current_images, action in tqdm(train_dataloader, desc="Validating", leave=False):

                    action = action.to(DEVICE)
                    output, _ = model(current_images)
                    prediction = torch.argmax(output, dim=2)

                    prediction_mask = torch.all(prediction == action, dim=1)
                    num_total += prediction_mask.shape[0]
                    num_correct += prediction_mask.sum().item()

            accuracy = (num_correct / num_total) * 100
            tqdm.write(f'Accuracy: {accuracy}')
            accuracies.append([epoch+1, accuracy])

            # Save parameters
            numpy.save(training_losses_path, training_losses)
            numpy.save(accuracies_path, accuracies)
            model.train()

    print('Finished Training !')
    
    if PARAM['Num_Epochs'] % PARAM['Weight_Saving_Step'] != 0:
        weight_save_path = os.path.join(weight_save_dir, f'{PARAM["Num_Epochs"]}.pth')
        torch.save(model.state_dict(), weight_save_path)
        print(f'Save Weight {PARAM["Num_Epochs"]}!')

        model.eval()
        with torch.no_grad():
            num_correct, num_total = 0, 0
            for current_images, action in tqdm(train_dataloader, desc="Validating", leave=False):

                action = action.to(DEVICE)
                output, _ = model(current_images)
                prediction = torch.argmax(output, dim=2)

                prediction_mask = torch.all(prediction == action, dim=1)
                num_total += prediction_mask.shape[0]
                num_correct += prediction_mask.sum().item()

        accuracy = (num_correct / num_total) * 100
        print(f'Accuracy: {accuracy}')
        accuracies.append([PARAM['Num_Epochs'], accuracy])

    # Plot and save graphs
    plot_graph(training_losses, accuracies, PARAM, end_plot=PARAM["Num_Epochs"], figure_path=result_path)

if __name__ == '__main__':

    model_name = 'DinoMlp5Bi'
    map_path = '/data/SPOT_Real_World_Dataset/map1'
    result_path = '/root/Object_Centric_Local_Navigation/training/results/DinoMlp5Bi'
    use_embedding = False

    import re, importlib
    
    module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
    module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
    module = importlib.import_module(module_path)
    if use_embedding:
        model = getattr(module, model_name)(use_gsam=False)
    else:
        model = getattr(module, model_name)()

    train_single_step(model, map_path, result_path, use_embedding=use_embedding)