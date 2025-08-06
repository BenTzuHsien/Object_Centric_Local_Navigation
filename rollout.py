import torch, os, numpy, time, yaml, select, termios, tty
from collections import OrderedDict
from SpotStack import GraphCore
from Object_Centric_Local_Navigation.object_centric_local_navigation import ObjectCentricLocalNavigation
from torchvision.transforms import ToPILImage

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr != []

class Rollout(ObjectCentricLocalNavigation):
    TRANSLATION_TOLERANCE = 0.2
    ROTATION_TOLERANCE = 0.09
    EVALUATION_DICT_ORDER = ["success", "duration", "translation_error", "rotation_error", "pose_error"]

    def __init__(self, architecture, weight_name, robot, eval_graph_path):
        super().__init__(architecture, weight_name, robot)
        
        self._graph_core = GraphCore(robot, eval_graph_path)
        self._graph_core.load_graph()

    def get_observation(self, step_dir):

        if not os.path.exists(step_dir):
            os.mkdir(step_dir)

        current_images = self._image_fetcher.get_images()
        observation = []

        for index, image in enumerate(current_images):
            # Save current images
            image_path = os.path.join(step_dir, f'{index}.jpg')
            image.save(image_path)

            if self.data_transforms:
                image = self.data_transforms(image)
            observation.append(image)
        observation = torch.stack(observation).unsqueeze(0).to('cuda')

        return observation
    
    def predict(self, observation):

        with torch.no_grad():
            output_logist, debug_info = self._model(observation)
            prediction = torch.argmax(output_logist, dim=2).flatten()

        return prediction, debug_info[0]
    
    def run(self, goal_images, prompt, traj_dir):

        if not os.path.exists(traj_dir):
            os.mkdir(traj_dir)

        # Get Goal Image
        if self.data_transforms:
            goal_images_transformed = []
            for image in goal_images:
                image = self.data_transforms(image)
                goal_images_transformed.append(image)
            goal_images = torch.stack(goal_images_transformed).to('cuda')

        self._model.set_goal(goal_images, prompt)

        success = False
        step = 0
        actions = []

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            start_time = time.time()
            while not success:

                if key_pressed():
                    print('Manully Stop Robot !')
                    self.stop()
                    break

                step_dir = os.path.join(traj_dir, f'{step:02}')
                observation = self.get_observation(step_dir)
                
                prediction, masks = self.predict(observation)

                # Save Segmentation
                segmentation_dir = os.path.join(step_dir, 'segmentation')
                os.makedirs(segmentation_dir, exist_ok=True)
                observation = observation.squeeze(0)
                for i in range(4):
                    if masks[i] is not None:
                        masked_image = observation[i] * masks[i]
                        pil_image = ToPILImage()(masked_image)

                        image_path = os.path.join(segmentation_dir, f'{i}.jpg')
                        pil_image.save(image_path)

                actions.append(prediction)
                print(prediction)
                
                success = self.move(prediction)
                step += 1

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)

        duration = time.time() - start_time

        # Save actions
        actions = torch.stack(actions)
        actions_path = os.path.join(traj_dir, 'actions.csv')
        numpy.savetxt(actions_path, actions.cpu().numpy(), delimiter=',', fmt='%d')

        # Evaluation
        evaluation_dict = {}
        evaluation_dict['duration'] = duration
        
        evaluation_dict['pose_error'] = self._graph_core.get_relative_pose_from_waypoint('Goal_Pose')
        evaluation_dict['translation_error'] = abs(((evaluation_dict['pose_error'].x ** 2) + (evaluation_dict['pose_error'].y ** 2)) ** 0.5)
        evaluation_dict['rotation_error'] = abs(evaluation_dict['pose_error'].rotation.to_yaw())

        if evaluation_dict['translation_error'] < self.TRANSLATION_TOLERANCE and evaluation_dict['rotation_error'] < self.ROTATION_TOLERANCE:
            print(f'Succeeded !!, duration: {duration}')
            evaluation_dict['success'] = True
        else:
            print(f'Failed !, duration: {duration}')
            evaluation_dict['success'] = False

        # Reorder evaluation_dict
        evaluation_dict = OrderedDict((k, evaluation_dict[k]) for k in self.EVALUATION_DICT_ORDER if k in evaluation_dict)
        evaluation_path = os.path.join(traj_dir, 'evaluation.yaml')
        with open(evaluation_path, 'w') as file:
            yaml.dump(evaluation_dict, file)

if __name__ == '__main__':

    MODEL = ''
    WEIGHT = ''

    # radii 1.0, 0.5
    radii = [0.5]
    angles = [80, 50, 25, 0, -25, -50, -80]
    orientations = [135, 90, 45, 0, -45, -90, -135]

    import argparse, bosdyn.client.util, sys
    from PIL import Image
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    from bosdyn.api.geometry_pb2 import Vec2, SE2Pose
    from SpotStack import GraphNavigator
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('Rollout')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    
    rollout_graph_path = options.graph_path
    goal_image_dir = os.path.join(rollout_graph_path, 'Goal_Images')
    goal_images = []
    for image_name in sorted(os.listdir(goal_image_dir)):
        image_path = os.path.join(goal_image_dir, image_name)
        image = Image.open(image_path)
        goal_images.append(image)
    
    target_txt_path = os.path.join(rollout_graph_path, 'target_object.txt')
    with open(target_txt_path, "r") as f:
        prompt = f.read().strip()
    print(f'Target Object: {prompt}')

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                rollout_model = Rollout(MODEL, WEIGHT, robot, rollout_graph_path)
                graph_navigator = GraphNavigator(robot, rollout_graph_path)
                
                traj_num = 7
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:
                            
                            traj_dir = os.path.join(rollout_graph_path, f'{traj_num:03}')
                            if not os.path.exists(traj_dir):
                                os.mkdir(traj_dir)

                            # Save the Starting Point Configuration
                            starting_point_config = {}
                            starting_point_config['radius'] = rad
                            starting_point_config['angle'] = ang
                            starting_point_config['orientation'] = ori
                            stating_point_config_path = os.path.join(traj_dir, 'stating_point_config.yaml')
                            with open(stating_point_config_path, 'w') as file:
                                yaml.dump(starting_point_config, file, sort_keys=False)

                            # Starting Point
                            orientation_in_radius = (ori / 180) * numpy.pi
                            starting_pose = SE2Pose(position=Vec2(x=x, y=y), angle=orientation_in_radius)
                            graph_navigator.navigate_to(f'Goal_Pose', starting_pose)

                            print(f'Starting Rollout {traj_num}')
                            rollout_model.run(goal_images, prompt, traj_dir)
                            traj_num += 1
                            time.sleep(1.5)

            except Exception as exc:  # pylint: disable=broad-except
                print("Rollout threw an error.")
                print(exc)
            finally:
                rollout_model.on_quit()

    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )