import os, torch, shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2

def extract_embeddings(dataset_dir):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    # Models
    dinov2 = DinoV2().to(device='cuda')
    for param in dinov2.parameters():
        param.requires_grad = False
    dinov2.eval()

    dinotxt = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l').to(device='cuda')
    for param in dinotxt.parameters():
        param.requires_grad = False
    dinotxt.eval()
    avg_pool = torch.nn.AdaptiveAvgPool2d((16, 16 * 4))

    embeddings_dir_name = f'{os.path.basename(dataset_dir)}_baseline'
    embeddings_dir = os.path.join(os.path.dirname(dataset_dir), embeddings_dir_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Copy target txt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    target_txt_save_path = os.path.join(embeddings_dir, 'target_object.txt')
    shutil.copy(target_txt_path, target_txt_save_path)

    # ----- Process Goal Images -----
    goal_images_dir = os.path.join(dataset_dir, 'Goal_Images')
    goal_images = []
    for i in range(4):
        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    goal_images = torch.stack(goal_images).to(device='cuda')
    N, C, H, W = goal_images.shape

    ## Encode goal images
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        goal_embeds = dinov2(goal_images)
    _, C_out, H_out, W_out = goal_embeds.shape
    goal_embeds = goal_embeds.permute(1, 2, 0, 3).reshape(C_out, H_out, N * W_out)

    ## Pooling
    goal_embeds = avg_pool(goal_embeds)
    goal_embedding = goal_embeds.flatten(start_dim=1).permute(1, 0)

    ## Save tensors
    goal_embedding_path = os.path.join(embeddings_dir, 'goal_embedding.pt')
    torch.save(goal_embedding, goal_embedding_path)

    # ----- Process Current Images -----
    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

    for trajectory in tqdm(trajectories, desc='Extracting'):
        trajectory_dir = os.path.join(dataset_dir, trajectory)
        
        trajectory_embeddings_dir = os.path.join(embeddings_dir, trajectory)
        os.makedirs(trajectory_embeddings_dir, exist_ok=True)

        # Copy labels
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        label_save_path = os.path.join(trajectory_embeddings_dir, 'actions.csv')
        shutil.copy(label_path, label_save_path)

        # Extract Steps
        traj_images = []
        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())
        for step in steps:
            step_dir = os.path.join(trajectory_dir, step)
            
            current_images = []
            for i in range(4):
                img_path = os.path.join(step_dir, f'{i}.jpg')
                image = Image.open(img_path)
                image_tensor = transform(image)
                current_images.append(image_tensor)
            current_images = torch.stack(current_images)

            traj_images.append(current_images)
        traj_images = torch.stack(traj_images).to(device='cuda')

        ## Encode current images
        batch_size = traj_images.shape[0]
        traj_images = traj_images.reshape(batch_size*N, C, H, W)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            traj_embeds = dinov2(traj_images)
        
        _, C_out, H_out, W_out = traj_embeds.shape
        traj_embeds = traj_embeds.reshape(batch_size, N, C_out, H_out, W_out)
        traj_embeds = traj_embeds.permute(0, 2, 3, 1, 4).reshape(batch_size, C_out, H_out, N * W_out)

        ## Pooling
        traj_embeds = avg_pool(traj_embeds)
        traj_embeddings = traj_embeds.flatten(start_dim=2).permute(0, 2, 1)

        ## Save tensors
        traj_embedding_path = os.path.join(trajectory_embeddings_dir, 'embeddings.pt')
        torch.save(traj_embeddings, traj_embedding_path)

if __name__ == '__main__':

    dataset_dir = '/data/SPOT_Real_World_Dataset/green_chair'

    extract_embeddings(dataset_dir)