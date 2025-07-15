import importlib, re, os, torch, time
from torchvision import transforms
from SpotStack import ImageFetcher, MotionController

class ObjectCentricLocalNavigation:
    data_transforms = transforms.Resize((224, 224))
    STOP_THRESHOLD = 2
    ACTION_LOOKUP = {0: -0.2, 1: 0.0, 2: 0.2}

    def __init__(self, architecture, weight_name, robot):

        # Setup Model
        module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', architecture).lower()
        module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
        module = importlib.import_module(module_path)
        self._model = getattr(module, architecture)()
        
        # Load Weight
        weight_path = os.path.join(os.path.dirname(__file__), 'weights', weight_name)
        self._model.load_weight(weight_path)
        
        self._model.cuda()
        self._model.eval()

        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)
        self._motion_controller = MotionController(robot)
        self._stop_prediction_count = 0

    def on_quit(self):
        self._motion_controller.on_quit()

    def stop(self):
        self._motion_controller.send_velocity_command(0, 0, 0, duration=2)
        time.sleep(1)

    def get_observation(self):
        observation = self._image_fetcher.get_images(self.data_transforms)

        return observation

    def predict(self, observation):

        with torch.no_grad():
            output_logist, _ = self._model([observation])
            prediction = torch.argmax(output_logist, dim=2).flatten()

        return prediction
    
    def move(self, prediction):

        if (prediction[0] == 1) and (prediction[1] == 1) and (prediction[2] == 1):
            self.stop()
            self._stop_prediction_count += 1
            if self._stop_prediction_count >= self.STOP_THRESHOLD:
                return True
            
        else:
            self._stop_prediction_count = 0

            d_x, d_y, d_yaw = [self.ACTION_LOOKUP[p.item()] for p in prediction]
            self._motion_controller.send_displacement_command(d_x, d_y, d_yaw)
            time.sleep(1)

            return False
    
    def run(self, goal_images, target_object):

        # Get Goal Image
        if self.data_transforms:
            goal_images_transformed = []
            for image in goal_images:
                image = self.data_transforms(image)
                goal_images_transformed.append(image)
            goal_images = goal_images_transformed

        self._model.set_goal(goal_images, target_object)

        success = False
        while not success:
            observation = self.get_observation()
            prediction = self.predict(observation)
            success = self.move(prediction)

# Example Usage
if __name__ == '__main__':

    MODEL = ''
    WEIGHT = ''

    import argparse, bosdyn.client.util, sys
    from PIL import Image
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('ObjectCentricLocalNavigation')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    graph_path = options.graph_path
    goal_image_dir = os.path.join(graph_path, 'Goal_Images')
    goal_images = []
    for image_name in sorted(os.listdir(goal_image_dir)):
        image_path = os.path.join(goal_image_dir, image_name)
        image = Image.open(image_path)
        goal_images.append(image)

    target_txt_path = os.path.join(graph_path, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    print(f'Target Object: {prompt}')

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                rollout_model = ObjectCentricLocalNavigation(MODEL, WEIGHT, robot)
                rollout_model.run(goal_images, prompt)

            except Exception as exc:  # pylint: disable=broad-except
                print("ObjectCentricLocalNavigation threw an error.")
                print(exc)
            finally:
                rollout_model.on_quit()


    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )