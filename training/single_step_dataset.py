import os, torch, numpy
from torch.utils.data import Dataset
from torchvision import transforms

class SingleStepDataset(Dataset):
    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])
    
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir

        # Load data
        ## Get goal prompt
        target_txt_path = os.path.join(self.dataset_dir, 'target_object.txt')
        with open(target_txt_path, "r") as f:
            self.prompt = f.read().strip()

        # Get goal box
        goal_box_path = os.path.join(self.dataset_dir, 'goal_box.pt')
        self.goal_box = torch.load(goal_box_path, map_location='cpu', weights_only=True)

        ## Get goal embeddings
        goal_embedding_path = os.path.join(self.dataset_dir, 'goal_embedding.pt')
        self.goal_embedding = torch.load(goal_embedding_path, map_location='cpu', weights_only=True)   # C', H', W'

        ## Load current embeddings
        self.current_boxes = []
        self.current_embeddings = []
        self.actions = []

        trajectories = sorted(item for item in os.listdir(dataset_dir) if item.isdigit())
        for trajectory in trajectories:
            trajectory_dir = os.path.join(dataset_dir, trajectory)

            # Get box
            traj_boxes_path = os.path.join(trajectory_dir, 'boxes.pt')
            traj_boxes = torch.load(traj_boxes_path, map_location='cpu', weights_only=True)   # S, 4
            self.current_boxes.append(traj_boxes)

            # Get embeddings
            traj_embeddings_path = os.path.join(trajectory_dir, 'embeddings.pt')
            traj_embeddings = torch.load(traj_embeddings_path, map_location='cpu', weights_only=True)   # S, C', H', W'
            self.current_embeddings.append(traj_embeddings)

            # Get actions
            actions_path = os.path.join(trajectory_dir, 'actions.csv')
            actions = numpy.loadtxt(actions_path, delimiter=' ')
            actions = torch.tensor(actions, dtype=torch.long)
            self.actions.append(actions)

            if not traj_embeddings.shape[0] == actions.shape[0]:
                raise ValueError(f"Data length not consistent: trajectory: {trajectory}, current_embeddings={traj_embeddings.shape[0]}, labels={actions.shape[0]}")

        # Create samples
        self.samples = []

        for traj_index in range(len(trajectories)):

            for step_index in range(len(self.current_embeddings[traj_index])):

                current_box = self.current_boxes[traj_index][step_index]
                current_embedding = self.current_embeddings[traj_index][step_index]
                action = self.actions[traj_index][step_index]

                for goal_index in range(step_index+1, len(self.current_embeddings[traj_index])):

                    goal_box = self.current_boxes[traj_index][goal_index]
                    goal_embedding = self.current_embeddings[traj_index][goal_index]
                    sample = [current_box, current_embedding, goal_box, goal_embedding, action, self.prompt]
                    self.samples.append(sample)

                # add the real goal
                sample = [current_box, current_embedding, self.goal_box, self.goal_embedding, action, self.prompt]
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        current_box = self.samples[idx][0]
        current_embedding = self.samples[idx][1]
        goal_box = self.samples[idx][2]
        goal_embedding = self.samples[idx][3]
        action = self.samples[idx][4]
        prompt = self.samples[idx][5]
        
        return current_box, current_embedding, goal_box, goal_embedding, action, prompt
