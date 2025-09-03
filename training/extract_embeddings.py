import os, torch, shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from Object_Centric_Local_Navigation.models.modules.utils import get_masked_region

def extract_embeddings(dataset_dir, vision_encoder, segmentation_model):

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    avg_pool = torch.nn.AdaptiveAvgPool2d((16, 16))

    embeddings_dir_name = f'{os.path.basename(dataset_dir)}_{vision_encoder.__class__.__name__.lower()}_embeddings'
    embeddings_dir = os.path.join(os.path.dirname(dataset_dir), embeddings_dir_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Load Goal Prompt
    target_txt_path = os.path.join(dataset_dir, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    prompt = [prompt]

    # Copy target txt
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
    goal_panoramic = goal_images.permute(1, 2, 0, 3).reshape(C, H, N * W)

    ## Encode goal images
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        goal_embeds = vision_encoder(goal_images)
    _, C_out, H_out, W_out = goal_embeds.shape
    goal_embeds = goal_embeds.permute(1, 2, 0, 3).reshape(C_out, H_out, N * W_out)

    ## Segment goal imgaes
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        goal_boxes, goal_masks = segmentation_model(goal_panoramic.unsqueeze(0), prompt)
    goal_box = goal_boxes.squeeze(0)

    embedding_masks = torch.nn.functional.interpolate(goal_masks, [H_out, N * W_out], mode="nearest")
    embedding_boxes = get_masked_region(embedding_masks)

    x1, y1, x2, y2 = embedding_boxes[0]
    masked_embeddings = embedding_masks[0][:, y1:y2+1, x1:x2+1] * goal_embeds[:, y1:y2+1, x1:x2+1]
    goal_embedding = avg_pool(masked_embeddings)

    ## Save tensors
    goal_box_path = os.path.join(embeddings_dir, 'goal_box.pt')
    torch.save(goal_box.float(), goal_box_path)
    goal_embedding_path = os.path.join(embeddings_dir, 'goal_embedding.pt')
    torch.save(goal_embedding.float(), goal_embedding_path)

    ## Save segmented images
    masked_goal_image = goal_masks * goal_panoramic
    masked_goal_image_path = os.path.join(embeddings_dir, 'masked_goal_image.jpg')
    save_image(masked_goal_image, masked_goal_image_path)

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
        traj_panoramics = []
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
            current_panoramic = current_images.permute(1, 2, 0, 3).reshape(C, H, N * W)

            traj_images.append(current_images)
            traj_panoramics.append(current_panoramic)
        traj_images = torch.stack(traj_images).to(device='cuda')
        traj_panoramics = torch.stack(traj_panoramics).to(device='cuda')

        ## Encode current images
        batch_size = traj_images.shape[0]
        traj_images = traj_images.reshape(batch_size*N, C, H, W)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            traj_embeds = vision_encoder(traj_images)
        
        _, C_out, H_out, W_out = traj_embeds.shape
        traj_embeds = traj_embeds.reshape(batch_size, N, C_out, H_out, W_out)
        traj_embeds = traj_embeds.permute(0, 2, 3, 1, 4).reshape(batch_size, C_out, H_out, N * W_out)

        ## Segment current images
        prompts = prompt * batch_size
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            traj_boxes, traj_masks = segmentation_model(traj_panoramics, prompts)

        embedding_masks = torch.nn.functional.interpolate(traj_masks, [H_out, N * W_out], mode="nearest")
        embedding_boxes = get_masked_region(embedding_masks)

        traj_embeddings = []
        for i in range(batch_size):
            x1, y1, x2, y2 = embedding_boxes[i]
            masked_embeddings = embedding_masks[i][:, y1:y2+1, x1:x2+1] * traj_embeds[i][:, y1:y2+1, x1:x2+1]
            masked_embeddings = avg_pool(masked_embeddings)
            traj_embeddings.append(masked_embeddings)
        traj_embeddings = torch.stack(traj_embeddings)

        ## Save tensors
        traj_box_path = os.path.join(trajectory_embeddings_dir, 'boxes.pt')
        torch.save(traj_boxes.float(), traj_box_path)
        traj_embedding_path = os.path.join(trajectory_embeddings_dir, 'embeddings.pt')
        torch.save(traj_embeddings.float(), traj_embedding_path)

        ## Save segmented images
        masked_traj_image = traj_masks * traj_panoramics
        masked_traj_image_path = os.path.join(trajectory_embeddings_dir, 'masked.jpg')
        save_image(masked_traj_image, masked_traj_image_path)

if __name__ == '__main__':

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2

    dataset_dir = ''
    vision_encoder = DinoV2().to(device='cuda')
    segmentation_model = OwlV2Sam2().to(device='cuda')

    extract_embeddings(dataset_dir, vision_encoder, segmentation_model)