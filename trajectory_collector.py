import os, time, numpy, shutil
from SpotStack import GraphNavigator, ImageFetcher

from bosdyn.client.exceptions import ResponseError

class TrajectoryCollector(GraphNavigator):
    RECORD_INTERVAL = 0.2
    DISCRETIZED_TOLERANCE = 0.02

    def __init__(self, robot, graph_path):
        super().__init__(robot, graph_path)

        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)
        self._traj_dir = None
        self._pre_pose = None
    
    def _record_images(self, step_num):

        step_dir = os.path.join(self._traj_dir, f'{step_num:02}')
        if not os.path.exists(step_dir):
            os.mkdir(step_dir)
        
        current_images = self._image_fetcher.get_images()
        
        for index, image in enumerate(current_images):
            image_path = os.path.join(step_dir, f'{index}.jpg')
            image.save(image_path)
        
        self._pre_pose = self.get_relative_pose_from_waypoint('Goal_Pose')

    def _get_displacement(self):

        current_pose = self.get_relative_pose_from_waypoint('Goal_Pose')
        displacement_pose = self._pre_pose.inverse() * current_pose
        displacement = (displacement_pose.x, displacement_pose.y, displacement_pose.rot.to_yaw())
        return displacement
    
    @classmethod
    def _discretize_action(cls, displacements):

        action = [1, 1, 1]
        for i in range(3):
            if displacements[i] > cls.DISCRETIZED_TOLERANCE:
                action[i] = 2
            elif displacements[i] < -cls.DISCRETIZED_TOLERANCE:
                action[i] = 0
        return action

    def navigate_and_record_to(self, waypoint_name, traj_dir):

        destination_waypoint = self._current_annotation_name_to_wp_id[waypoint_name]
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self._power_manager.toggle_power(should_power_on=True):
            print("GraphNavigator: Failed to power on the robot, and cannot complete navigate to request.")
            return
        
        # Setup for this trajectory
        nav_to_cmd_id = None
        is_finished = False
        if not os.path.exists(traj_dir):
            os.mkdir(traj_dir)
        self._traj_dir = traj_dir
        action_index = 0
        actions = numpy.empty([0, 3])

        # Navigate to the destination waypoint.
        self._record_images(action_index)
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)

            except ResponseError as e:
                print(f"GraphNavigator: Error while navigating {e}")
                break
            
            time.sleep(self.RECORD_INTERVAL)
            displacements = self._get_displacement()
            if abs(displacements[0]) < self.DISCRETIZED_TOLERANCE and abs(displacements[1]) < self.DISCRETIZED_TOLERANCE and abs(displacements[2]) < self.DISCRETIZED_TOLERANCE:
                shutil.rmtree(os.path.join(self._traj_dir, f'{action_index:02}'))
            else:
                action = self._discretize_action(displacements)
                actions = numpy.vstack([actions, action])
                action_index += 1
            
            self._record_images(action_index)
            
            # Poll the robot for feedback to determine if the navigation command is complete.
            is_finished = self._check_success(nav_to_cmd_id)

        time.sleep(self.RECORD_INTERVAL)
        action = [1, 1, 1]
        actions = numpy.vstack([actions, action])
        actions_path = os.path.join(self._traj_dir, 'actions.csv')
        numpy.savetxt(actions_path, actions, fmt='%d')

if __name__ == '__main__':

    radii = [0.8, 0.4]
    angles = [30, 0, -30]
    orientations = [45, 0, -45]

    import argparse, bosdyn.client.util, sys
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
    from bosdyn.api.geometry_pb2 import Vec2, SE2Pose
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('TrajectoryCollector')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    graph_path = options.graph_path
    goal_image_path = os.path.join(graph_path, 'Goal_Image.jpg')

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                trajectory_collector = TrajectoryCollector(robot, graph_path)
                
                traj_num = 0
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:
                            
                            traj_dir = os.path.join(graph_path, f'{traj_num:02}')

                            # Starting Point
                            orientation_in_radius = (ori / 180) * numpy.pi
                            starting_pose = SE2Pose(position=Vec2(x=x, y=y), angle=orientation_in_radius)
                            trajectory_collector.navigate_to(f'Goal_Pose', starting_pose)

                            # Record
                            print(f'Starting Trajectory {traj_num}')
                            trajectory_collector.navigate_and_record_to('Goal_Pose', traj_dir)
                            print(f'Finished Trajectory {traj_num}')
                            traj_num += 1
                            time.sleep(1.5)

            except Exception as exc:  # pylint: disable=broad-except
                print("TrajectoryCollector threw an error.")
                print(exc)
            finally:
                trajectory_collector.on_quit()

    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )