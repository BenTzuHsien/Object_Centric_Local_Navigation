import os, torch, shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from Object_Centric_Local_Navigation.models.backbones.grounded_sam2 import GroundedSAM2

def extract_backbone_features(backbone, dataset_dir, destination_dir):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)

    # Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompts = [prompt] * 4

    # Copy target txt
    target_txt_save_path = os.path.join(destination_dir, 'target_object.txt')
    shutil.copy(target_txt_path, target_txt_save_path)

    # Process Goal Images
    goal_image_dir = os.path.join(dataset_dir, 'Goal_Images')
    goal_images = []
    for i in range(4):
        goal_image = Image.open(os.path.join(goal_image_dir, f'{i}.jpg'))
        goal_image_tensor = transform(goal_image)
        goal_images.append(goal_image_tensor)
    goal_images = torch.stack(goal_images).to('cuda')
    goal_embeddings, _ = backbone(goal_images, prompts)
    goal_embeddings = goal_embeddings.unsqueeze(0)   #1, 4, C, H, W

    goal_embeddings_path = os.path.join(destination_dir, 'goal_embeddings.pt')
    torch.save(goal_embeddings, goal_embeddings_path)

    # Process Current Images
    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

    for trajectory in tqdm(trajectories, desc='Extracting'):
        trajectory_dir = os.path.join(dataset_dir, trajectory)
        
        trajectory_save_dir = os.path.join(destination_dir, trajectory)
        if not os.path.exists(trajectory_save_dir):
            os.mkdir(trajectory_save_dir)

        # Copy labels
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        label_save_path = os.path.join(trajectory_save_dir, 'actions.csv')
        shutil.copy(label_path, label_save_path)

        # Extract Steps
        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())
        for step in steps:
            step_dir = os.path.join(trajectory_dir, step)
            
            current_images = []
            for i in range(4):
                img_path = os.path.join(step_dir, f'{i}.jpg')
                image = Image.open(img_path)
                image_tensor = transform(image)
                current_images.append(image_tensor)
            current_images = torch.stack(current_images).to('cuda')
            current_embeddings, _ = backbone(current_images, prompts)   # 4, C, H, W
            
            step_save_path = os.path.join(trajectory_save_dir, f'{step}.pt')
            torch.save(current_embeddings, step_save_path)

if __name__ == '__main__':

    dataset_dir = ''
    destination_dir = ''

    gsam = GroundedSAM2().to('cuda')
    extract_backbone_features(gsam, dataset_dir, destination_dir)