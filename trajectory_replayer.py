import numpy
from SpotStack import MotionController
from object_centric_local_navigation import ObjectCentricLocalNavigation

class TrajectoryReplayer(ObjectCentricLocalNavigation):
    STOP_THRESHOLD = 0

    def __init__(self, robot):
        self._motion_controller = MotionController(robot)

    def run(self, actions_path):

        actions = numpy.loadtxt(actions_path, delimiter=' ')
        for action in actions:
            self.move(action)

# Example Usage
if __name__ == '__main__':

    radii = [0.4, 0.5, 0.6, 0.9, 1.2]
    angles = [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90]
    orientations = [150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150]

    import argparse, bosdyn.client.util, os, sys, time
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
    sdk = bosdyn.client.create_standard_sdk('TrajectoryReplayer')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                
                trajectory_replayer = TrajectoryReplayer(robot)
                graph_navigator = GraphNavigator(robot, options.graph_path)

                traj_num = 0
                for rad in radii:
                    for ang in angles:
                        angle_in_radius = (ang / 180) * numpy.pi
                        x = -rad * numpy.cos(angle_in_radius)
                        y = rad * numpy.sin(angle_in_radius)
                        
                        for ori in orientations:

                            actions_path = os.path.join(options.graph_path, f'{traj_num:03}', 'actions.csv')

                            # Starting Point
                            orientation_in_radius = (ori / 180) * numpy.pi
                            starting_pose = SE2Pose(position=Vec2(x=x, y=y), angle=orientation_in_radius)
                            graph_navigator.navigate_to(f'Goal_Pose', starting_pose)

                            # Replay
                            print(f'Starting Trajectory {traj_num}')
                            trajectory_replayer.run(actions_path)
                            print(f'Finished Trajectory {traj_num}')
                            traj_num += 1
                            time.sleep(1.5)

            except Exception as exc:  # pylint: disable=broad-except
                print("TrajectoryReplayer threw an error.")
                print(exc)
            finally:
                trajectory_replayer.on_quit()


    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )