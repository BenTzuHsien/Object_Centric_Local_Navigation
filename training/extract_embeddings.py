import os, torch, shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def extract_embeddings(dataset_dir, vision_encoder, segmentation_model):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    embeddings_dir_name = f'{os.path.basename(dataset_dir)}_{vision_encoder.__class__.__name__.lower()}_embeddings'
    embeddings_dir = os.path.join(os.path.dirname(dataset_dir), embeddings_dir_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    segmentation_dir_name = f'{os.path.basename(dataset_dir)}_segmentation'
    segmentation_dir = os.path.join(os.path.dirname(dataset_dir), segmentation_dir_name)
    os.makedirs(segmentation_dir, exist_ok=True)

    # Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompts = [prompt] * 4

    # Copy target txt
    target_txt_save_path = os.path.join(embeddings_dir, 'target_object.txt')
    shutil.copy(target_txt_path, target_txt_save_path)

    # Process Goal Images
    goal_images_dir = os.path.join(dataset_dir, 'Goal_Images')
    goal_images = []
    for i in range(4):
        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    goal_images = torch.stack(goal_images).to(device='cuda')

    goal_embeds = vision_encoder(goal_images)
    goal_embeddings, goal_masks = segmentation_model(goal_images, prompts, goal_embeds)

    _, C_out, H_out, W_out = goal_embeddings.shape
    goal_embeddings = goal_embeddings.permute(1, 2, 0, 3).reshape(C_out, H_out, goal_images.shape[0] * W_out)   # Concatenate the images horizontally
    goal_embeddings_path = os.path.join(embeddings_dir, 'goal_embeddings.pt')
    torch.save(goal_embeddings, goal_embeddings_path)

    masked_goal_images = []
    for i in range(4):
        if goal_masks[i] is not None:
            masked_image = goal_images[i] * goal_masks[i]
        else:
            masked_image = torch.zeros_like(goal_images[i])
        masked_goal_images.append(masked_image)
    masked_goal_images = torch.cat(masked_goal_images, dim=2)
    masked_goal_image_path = os.path.join(segmentation_dir, 'masked_goal_image.jpg')
    save_image(masked_goal_images, masked_goal_image_path)

    # Process Current Images
    trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

    for trajectory in tqdm(trajectories, desc='Extracting'):
        trajectory_dir = os.path.join(dataset_dir, trajectory)
        
        trajectory_embeddings_dir = os.path.join(embeddings_dir, trajectory)
        os.makedirs(trajectory_embeddings_dir, exist_ok=True)

        trajectory_segmentation_dir = os.path.join(segmentation_dir, trajectory)
        os.makedirs(trajectory_segmentation_dir, exist_ok=True)

        # Copy labels
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        label_save_path = os.path.join(trajectory_embeddings_dir, 'actions.csv')
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

            current_images = torch.stack(current_images).to(device='cuda')
            current_embeds = vision_encoder(current_images)
            current_embeddings, masks = segmentation_model(current_images, prompts, current_embeds)
            
            _, C_out, H_out, W_out = current_embeddings.shape
            current_embeddings = current_embeddings.reshape(1, 4, C_out, H_out, W_out).squeeze(0)
            current_embeddings = current_embeddings.permute(1, 2, 0, 3).reshape(C_out, H_out, 4 * W_out)   # Concatenate the images horizontally
            step_embeddings_save_path = os.path.join(trajectory_embeddings_dir, f'{step}.pt')
            torch.save(current_embeddings, step_embeddings_save_path)
            
            masked_current_images = []
            for i in range(4):
                if masks[i] is not None:
                    masked_image = current_images[i] * masks[i]
                else:
                    masked_image = torch.zeros_like(current_images[i])
                
                masked_current_images.append(masked_image)
            masked_current_images = torch.cat(masked_current_images, dim=2)
            step_segmentation_save_path = os.path.join(trajectory_segmentation_dir, f'{step}.jpg')
            save_image(masked_current_images, step_segmentation_save_path)

if __name__ == '__main__':

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2

    dataset_dir = ''
    vision_encoder = DinoV2().to(device='cuda')
    segmentation_model = OwlV2Sam2().to(device='cuda')

    extract_embeddings(dataset_dir, vision_encoder, segmentation_model)