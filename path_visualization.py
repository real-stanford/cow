import argparse
import json
import os

from ai2thor.controller import Controller
from PIL import Image

from src.simulation.visualization_utils import get_agent_map_data, visualize_agent_path


def main(args):
    top_height = 860
    top_width = 860

    ego_height = 480
    ego_width = 640

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    controller = Controller(
        local_executable_path=args.thor_build,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        agentType="stochastic",
        continuousMode=True,
        snapToGrid=False,
        agentMode="locobot",
        fieldOfView=63.453048374758716,
        scene=args.thor_floor,
        height=ego_height,
        width=ego_width,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )

    data = None
    with open(args.result_json, "r") as f:
        data = json.load(f)

    traj = data["episode_metrics"]["trajectory"]

    for i in range(len(traj)):
        teleport_action = {
            "action": "TeleportFull",
            "position": {"x": traj[i]["x"], "y": traj[i]["y"], "z": traj[i]["z"]},
            "rotation": {"x": 0, "y": traj[i]["rotation"], "z": 0},
            "horizon": 0,
        }
        event = controller.step(action=teleport_action)
        Image.fromarray(event.frame).save(f"{args.out_dir}/ego_{i}.png")

    controller.reset(height=top_height, width=top_width)

    for i in range(len(traj)):
        teleport_action = {
            "action": "TeleportFull",
            "position": {"x": traj[i]["x"], "y": traj[i]["y"], "z": traj[i]["z"]},
            "rotation": {"x": 0, "y": traj[i]["rotation"], "z": 0},
            "horizon": 0,
        }
        controller.step(action=teleport_action)
        meta = get_agent_map_data(controller)
        frame = visualize_agent_path(
            traj[: i + 1],
            meta["frame"],
            meta["pos_translator"],
            color_pair_ind=2,
            opacity=0,
            only_show_last_visibility_cone=True,
            max_colors=len(traj),
        )

        img = Image.fromarray(frame)
        img = img.crop((26, 307, 807, 656))
        img.save(f"{args.out_dir}/top_{i}.png")

    os.system(
        f"ffmpeg -framerate 5 -i {args.out_dir}/ego_%01d.png -vcodec mpeg4 -vb 20M -y {args.out_dir}/all_ego.mp4"
    )
    os.system(
        f"ffmpeg -framerate 5 -i {args.out_dir}/top_%01d.png -vcodec mpeg4 -vb 20M -y {args.out_dir}/all_top.mp4"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggrigate results.")

    parser.add_argument(
        "--out-dir",
        action="store",
        type=str,
        required=True,
        help="e.g., viz/ an output dir to store the visualization",
    )
    parser.add_argument(
        "--thor-floor",
        action="store",
        type=str,
        required=True,
        help="e.g., FloorPlan_Val3_5, which is a thor scene name",
    )
    parser.add_argument(
        "--result-json",
        action="store",
        type=str,
        required=True,
        help="e.g., media/media_data/FloorPlan_Val3_5_GingerbreadHouse_1.json or any other evaluation",
    )
    parser.add_argument(
        "--thor-build",
        action="store",
        type=str,
        required=True,
        help="e.g., pasture_builds/thor_build_longtail/longtail.x86_64",
    )

    args = parser.parse_args()
    main(args)
