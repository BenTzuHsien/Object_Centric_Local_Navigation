import os
from SpotStack import GraphRecorder, ImageFetcher

class GetGoalCondition:

    def __init__(self, robot, graph_path):
        self.graph_path = graph_path
        self._graph_recorder = GraphRecorder(robot, graph_path)
        self._image_fetcher = ImageFetcher(robot, use_front_stitching=True)

        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

    def record_goal(self):

        self._graph_recorder.start_recording()
        self._graph_recorder.record_waypoint(f'Goal_Pose')

        current_images = self._image_fetcher.get_images()
        
        front_image_path = os.path.join(self.graph_path, 'Goal_Image.jpg')
        current_images[0].save(front_image_path)

        self._graph_recorder.stop_recording()
        self._graph_recorder.download_full_graph()
        print('Goal Pose Recorded !')

if __name__ == '__main__':

    import argparse, bosdyn.client.util, sys
    
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--graph-path',
                        help='Full filepath for the graph.',
                        default=os.getcwd())
    options = parser.parse_args(sys.argv[1:])

    # Create robot object
    sdk = bosdyn.client.create_standard_sdk('GetGoalCondition')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    get_goal_condition = GetGoalCondition(robot, options.graph_path)

    try:
        get_goal_condition.record_goal()

    except Exception as exc:  # pylint: disable=broad-except
        print("GetGoalCondition threw an error.")
        print(exc)