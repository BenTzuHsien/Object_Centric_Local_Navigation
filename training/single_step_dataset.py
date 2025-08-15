import os, torch, numpy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SingleStepDataset(Dataset):
    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    def __init__(self, dataset_dir, use_embeddings=False, pre_load=False):
        self.use_embeddings = use_embeddings
        self.pre_load = pre_load

        self.dataset_dir = dataset_dir
        self.labels = torch.empty([0, 3], dtype=torch.long)
        trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())

        if self.use_embeddings is False:
            if self.pre_load:
                self.current_images = []

                for trajectory in trajectories:
                    trajectory_dir = os.path.join(dataset_dir, trajectory)

                    # Current Images
                    traj_imgs = self._load_trajectory_images(trajectory_dir)
                    self.current_images.extend(traj_imgs)

                    # Labels
                    traj_labels = self._load_trajectory_labels(trajectory_dir)
                    self.labels = torch.vstack([self.labels, traj_labels])

                self.current_images = torch.stack(self.current_images).share_memory_()   # S, N, C, H, W
                self.labels = self.labels.share_memory_()
                if not self.current_images.shape[0] == self.labels.shape[0]:
                    raise ValueError(f"Data length not consistent: current_images={len(self.current_images_paths)}, labels={self.labels.shape[0]}")
                else:
                    self._len = self.current_images.shape[0]

            else:
                self.current_images_paths = []

                for trajectory in trajectories:
                    trajectory_dir = os.path.join(dataset_dir, trajectory)

                    # Current Images
                    traj_imgs_paths = self._load_trajectory_images_paths(trajectory_dir)
                    self.current_images_paths.extend(traj_imgs_paths)

                    # Labels
                    traj_labels = self._load_trajectory_labels(trajectory_dir)
                    self.labels = torch.vstack([self.labels, traj_labels])

                self.labels = self.labels.share_memory_()
                if not len(self.current_images_paths) == self.labels.shape[0]:
                    raise ValueError(f"Data length not consistent: current_images_paths={len(self.current_images_paths)}, labels={self.labels.shape[0]}")
                else:
                    self._len = len(self.current_images_paths)

        else:
            current_embeddings = []

            for trajectory in trajectories:
                trajectory_dir = os.path.join(dataset_dir, trajectory)

                # Current Embeddings
                traj_embeddings = self._load_trajectory_embeddings(trajectory_dir)
                current_embeddings.append(traj_embeddings)

                # Labels
                traj_labels = self._load_trajectory_labels(trajectory_dir)
                self.labels = torch.vstack([self.labels, traj_labels])

            self.current_embeddings = torch.cat(current_embeddings).share_memory_()   # S, C, H, W
            self.labels = self.labels.share_memory_()
            if not self.current_embeddings.shape[0] == self.labels.shape[0]:
                raise ValueError(f"Data length not consistent: current_embeddings={len(self.current_embeddings.shape[0])}, labels={self.labels.shape[0]}")
            else:
                self._len = self.current_embeddings.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        if not self.use_embeddings:
            if self.pre_load:
                return self.current_images[idx], self.labels[idx]
            
            else:
                step_imgs = self._load_step_images(self.current_images_paths[idx])
                return step_imgs, self.labels[idx]
        else:
            return self.current_embeddings[idx], self.labels[idx]
    
    def get_goal(self):

        # Goal Prompt
        target_txt_path = os.path.join(self.dataset_dir, 'target_object.txt')
        with open(target_txt_path, "r") as f:
            prompt = f.read().strip()

        # Goal Images
        if not self.use_embeddings:
            goal_images_dir = os.path.join(self.dataset_dir, 'Goal_Images')
            goal_images = []
            for i in range(4):
                goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
                goal_image = self.transform(goal_image)
                goal_images.append(goal_image)
            goal_images = torch.stack(goal_images)
            return goal_images, prompt

        else:
            goal_embeddings_path = os.path.join(self.dataset_dir, 'goal_embeddings.pt')
            goal_embeddings = torch.load(goal_embeddings_path, map_location='cpu', weights_only=True)   # C, H, W
            return goal_embeddings, prompt
    
    @staticmethod
    def _load_trajectory_labels(trajectory_dir):
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        labels = numpy.loadtxt(label_path, delimiter=' ')
        traj_labels = torch.tensor(labels, dtype=torch.long)

        return traj_labels

    @staticmethod
    def _load_trajectory_images_paths(trajectory_dir):

        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())
        traj_imgs_paths = []
        for step in steps:
            step_dir = os.path.join(trajectory_dir, step)
            step_imgs_paths = []

            for i in range(4):
                img_path = os.path.join(step_dir, f'{i}.jpg')
                step_imgs_paths.append(img_path)
            
            traj_imgs_paths.append(step_imgs_paths)

        return traj_imgs_paths
    
    @staticmethod
    def _load_trajectory_embeddings(trajectory_dir):

        steps = sorted(x for x in os.listdir(trajectory_dir) if os.path.splitext(x)[0].isdigit())
        traj_embeddings = []
        for step in steps:
            step_embeddings_path = os.path.join(trajectory_dir, step)
            step_embeddings = torch.load(step_embeddings_path, map_location='cpu', weights_only=True)   # C, H, W
            traj_embeddings.append(step_embeddings)

        return torch.stack(traj_embeddings)   # S, C, H, W

    def _load_trajectory_images(self, trajectory_dir):

        steps = sorted(x for x in os.listdir(trajectory_dir) if x.isdigit())
        traj_images = []
        for step in steps:
            step_dir = os.path.join(trajectory_dir, step)
            step_images = []

            for i in range(4):
                step_img = Image.open(os.path.join(step_dir, f'{i}.jpg'))
                step_img = self.transform(step_img)
                step_images.append(step_img)

            step_images = torch.stack(step_images)
            traj_images.append(step_images)

        return traj_images   # [N, C, H, W] * S

    def _load_step_images(self, step_image_paths):
        step_imgs = []
        for img_path in step_image_paths:
            img = Image.open(img_path)
            img = self.transform(img)
            step_imgs.append(img)
        step_imgs = torch.stack(step_imgs)
        return step_imgs
