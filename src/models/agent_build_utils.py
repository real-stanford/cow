import json

from src.simulation.constants import (AGENT_HEIGHT_HABITAT_M,
                                      AGENT_HEIGHT_THOR_M,
                                      FLOOR_TOLERANCE_HABITAT_M,
                                      FLOOR_TOLERANCE_THOR_M,
                                      GPT_HABITAT_OBJECT_TYPES_CLIP,
                                      GPT_THOR_OBJECT_TYPES_CLIP,
                                      HABITAT_OBJECT_TYPES,
                                      HABITAT_OBJECT_TYPES_CLIP,
                                      THOR_LONGTAIL_OBJECT_TYPES_CLIP,
                                      THOR_LONGTAIL_TYPES, THOR_OBJECT_TYPES,
                                      THOR_OBJECT_TYPES_CLIP)
from src.simulation.sim_enums import ClassTypes, EnvTypes


def get_env_class_vars(prompts_path, env_type, class_type):
    classes = None
    classes_clip = None
    agent_height = None
    floor_tolerance = None
    negate_action = None

    prompts = None
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    if env_type == EnvTypes.HABITAT:

        classes = HABITAT_OBJECT_TYPES

        if class_type == ClassTypes.REGULAR:
            classes_clip = HABITAT_OBJECT_TYPES_CLIP
        elif class_type == ClassTypes.GPT:
            classes_clip = GPT_HABITAT_OBJECT_TYPES_CLIP
        else:
            raise ValueError(f'unsupported class_type: {class_type}')

        agent_height = AGENT_HEIGHT_HABITAT_M
        floor_tolerance = FLOOR_TOLERANCE_HABITAT_M
        negate_action = False
    else:
        classes = THOR_OBJECT_TYPES

        if class_type == ClassTypes.REGULAR or class_type == ClassTypes.SPATIAL or class_type == ClassTypes.APPEARENCE or class_type == ClassTypes.HIDDEN:
            classes_clip = THOR_OBJECT_TYPES_CLIP
        elif class_type == ClassTypes.GPT:
            classes_clip = GPT_THOR_OBJECT_TYPES_CLIP
        elif class_type == ClassTypes.LONGTAIL:
            classes = THOR_LONGTAIL_TYPES
            classes_clip = THOR_LONGTAIL_OBJECT_TYPES_CLIP

        agent_height = AGENT_HEIGHT_THOR_M
        floor_tolerance = FLOOR_TOLERANCE_THOR_M
        negate_action = False

    return classes, classes_clip, agent_height, floor_tolerance, negate_action, prompts
