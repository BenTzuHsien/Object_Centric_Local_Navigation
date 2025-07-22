import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Object_Centric_Local_Navigation.training.utils import get_least_used_gpu, get_top_available_gpus
from Object_Centric_Local_Navigation.training.single_step_dataset import SingleStepDataset

def validate_single_step(model, weight_path, dataset_path, num_gpus=1):
    
    # Setup model and device
    if num_gpus > 1:
        top_gpus = get_top_available_gpus(num_gpus)
        primary_device = f'cuda:{top_gpus[0]}'
        model = model.to(primary_device)
        model = torch.nn.DataParallel(model, device_ids=top_gpus)
        DEVICE = primary_device
    else:
        least_used_gpu = get_least_used_gpu()
        DEVICE = f'cuda:{least_used_gpu}'
        model = model.to(DEVICE)

    model.load_weight(weight_path)
    model.eval()

    # Setup dataset
    dataset = SingleStepDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    goal_images, prompt = dataset.get_goal()
    goal_images = goal_images.to(DEVICE)
    model.set_goal(goal_images, prompt)

    with torch.no_grad():
        num_correct, num_total = 0, 0
        for current_images, action in tqdm(dataloader, desc="Validating"):

            current_images = current_images.to(DEVICE)
            action = action.to(DEVICE)
            output, _ = model(current_images)
            prediction = torch.argmax(output, dim=2)

            prediction_mask = torch.all(prediction == action, dim=1)
            num_total += prediction_mask.shape[0]
            num_correct += prediction_mask.sum().item()

    accuracy = (num_correct / num_total) * 100
    print(f'Accuracy: {accuracy}')
    return accuracy

if __name__ == '__main__':

    model_name = ''
    weight_path = ''
    map_path = ''

    import re, importlib
    
    module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
    module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
    module = importlib.import_module(module_path)
    model = getattr(module, model_name)()

    validate_single_step(model, weight_path, map_path)