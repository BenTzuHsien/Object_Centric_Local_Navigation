import importlib, re, os, torch, time
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from SpotStack import ImageFetcher, MotionController

class ObjectCentricLocalNavigation:
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    STOP_THRESHOLD = 7
    ACTION_LOOKUP = {0: -0.2, 1: 0.0, 2: 0.2}

    def __init__(self, architecture, weight_name, robot):

        # Setup Model
        module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', architecture).lower()
        module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
        module = importlib.import_module(module_path)
        self._model = getattr(module, architecture)()

        # Load Weight
        weight_path = os.path.join(os.path.dirname(__file__), 'weights', weight_name)
        state_dict = torch.load(weight_path, map_location='cuda')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        self._model.load_state_dict(new_state_dict)

        self._model.cuda()
        self._model.eval()
        self._stop_prediction_count = 0
        
        self._image_fetcher = ImageFetcher(robot, use_front_stitching=False)
        self._motion_controller = MotionController(robot)

    def on_quit(self):
        self._motion_controller.on_quit()

    def get_observation(self):
        observation = self._image_fetcher.get_images(self.data_transforms)
        observation = observation.cuda().to(dtype=torch.float32)

        return observation

    def predict(self, observation):

        with torch.no_grad():
            output_logist = self._model(observation, self._goal_image)
            prediction = torch.argmax(output_logist, dim=2).flatten()

        return prediction
    
    def move(self, prediction):

        if (prediction[0] == 1) and (prediction[1] == 1) and (prediction[2] == 1):
            self._stop_prediction_count += 1
            if self._stop_prediction_count > self.STOP_THRESHOLD:
                return True
            
        else:
            self._stop_prediction_count = 0

            d_x, d_y, d_yaw = [self.ACTION_LOOKUP[p.item()] for p in prediction]
            self._motion_controller.send_displacement_command(d_x, d_y, d_yaw)
            time.sleep(1)

            return False
        
    def step(self):

        observation = self.get_observation()
        prediction = self.predict(observation)
        print(prediction)
        return self.move(prediction)
    
    def run(self, goal_image_path):

        # Get Goal Image
        goal_image = Image.open(goal_image_path)
        goal_image = self.data_transforms(goal_image).unsqueeze(0)
        self._goal_image = goal_image.cuda().to(dtype=torch.float32)

        success = False
        while not success:
            success = self.step()

# Example Usage
if __name__ == '__main__':

    import argparse, bosdyn.client.util, sys
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('ObjectCentricLocalNavigation')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    goal_image_path = '/home/ben/repo/Object_Centric_Local_Navigation/rollout/Goal_Image.jpg'

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                rollout_model = ObjectCentricLocalNavigation('DinoMlp5', 'DinoMLP5_discretized.pth', robot)
                prediction = rollout_model.run(goal_image_path)
                print(prediction)

                rollout_model.on_quit()

            except Exception as exc:  # pylint: disable=broad-except
                print("ObjectCentricLocalNavigation threw an error.")
                print(exc)

    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )