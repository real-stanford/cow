import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

TRAIN_ROOM_IDS = [f"_Train{i}_{j}" for i in range(1, 13) for j in range(1, 6)]
VAL_ROOM_IDS = [f"_Val{i}_{j}" for i in range(1, 4) for j in range(1, 6)]
TEST_ROOM_IDS = [
    f"_test-challenge{i}_{j}" for i in range(1, 6) for j in range(1, 3)]

ROBO_THOR_COMMIT_ID = "bad5bc2b250615cb766ffb45d455c211329af17e"
COW_ROBO_THOR_COMMIT_ID = "7676ca15870d6c9795be1ff1e7c12f65dd52474f"

IMAGE_WIDTH = 672
IMAGE_HEIGHT = 672
FOV = 90

AGENT_HEIGHT_THOR_M = 0.8658
AGENT_HEIGHT_HABITAT_M = 1.0
FLOOR_TOLERANCE_THOR_M = 0.05
FLOOR_TOLERANCE_HABITAT_M = 0.15

# NOTE: uncomment for fig
# VOXEL_SIZE_M = 0.08
VOXEL_SIZE_M = 0.125

# NOTE: uncomment for fig
# IN_CSPACE = False
IN_CSPACE = True

ROTATION_DEG = 30
FORWARD_M = 0.25
MAX_CEILING_HEIGHT_M = 1.9
TARGET_OBJECT_FOUND_M = 1.0
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLIP_MODEL_TO_FEATURE_DIM = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
}

PATCH_TO_ACTION_THOR = {
    0: "RotateLeft", 3: "RotateLeft", 6: "RotateLeft",
    1: "MoveAhead", 4: "MoveAhead", 7: "MoveAhead",
    2: "RotateRight", 5: "RotateRight", 8: "RotateRight",
}

ACTION_TO_PATCHES_THOR = {
    "RotateLeft": [0, 3, 6],
    "MoveAhead": [1, 4, 7],
    "RotateRight": [2, 5, 8]
}

ACTION_NEGATION = {
    "RotateLeft": "RotateRight",
    "RotateRight": "RotateLeft",
    "MoveAhead": "MoveBack",
    "MoveBack": "MoveAhead",
    "LookUp": "LookDown",
    "LookDown": "LookUp"
}

# NOTE: order important here as we want to expand least # of actions first
ACTION_SEQUENCES = (
    (("MoveAhead",), (0, FORWARD_M)),
    (("RotateLeft", "MoveAhead",), (ROTATION_DEG, FORWARD_M)),
    (("RotateRight", "MoveAhead",), (-ROTATION_DEG, FORWARD_M)),
    (("RotateLeft", "RotateLeft", "MoveAhead",), (ROTATION_DEG * 2, FORWARD_M)),
    (("RotateRight", "RotateRight", "MoveAhead",), (-ROTATION_DEG * 2, FORWARD_M)),
    (("RotateLeft", "RotateLeft", "RotateLeft", "MoveAhead",), (ROTATION_DEG * 3, FORWARD_M)),
    (("RotateRight", "RotateRight", "RotateRight", "MoveAhead",), (-ROTATION_DEG * 3, FORWARD_M)),
    (("RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "MoveAhead",), (ROTATION_DEG * 4, FORWARD_M)),
    (("RotateRight", "RotateRight", "RotateRight", "RotateRight", "MoveAhead",), (-ROTATION_DEG * 4, FORWARD_M)),
    (("RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "MoveAhead",), (ROTATION_DEG*5, FORWARD_M)),
    (("RotateRight", "RotateRight", "RotateRight", "RotateRight", "RotateRight", "MoveAhead",), (-ROTATION_DEG * 5, FORWARD_M)),
    (("RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "RotateLeft", "MoveAhead",), (ROTATION_DEG*6, FORWARD_M)),
)

ROTATION_MATRICIES = [R.from_euler("y", a[-1][0], degrees=True).as_matrix() for a in ACTION_SEQUENCES]

RENDERING_BOX_FRAC_THRESHOLD = 0.004

DATASET_KEYS = [
    'id',
    'initial_horizon',
    'initial_orientation',
    'initial_position',
    'object_type',
    'scene',
    'shortest_path',
    'shortest_path_length'
]

THOR_OBJECT_TYPES = [
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Bowl",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug",
    "SprayBottle",
    "Television",
    "Vase",
]

THOR_LONGTAIL_TYPES = [
    "GingerbreadHouse",
    "EspressoMachine",
    "Crate",
    "ElectricGuitar",
    "RiceCooker",
    "LlamaWickerBasket",
    "Whiteboard",
    "Surfboard",
    "Tricycle",
    "GraphicsCard",
    "Mate",
    "ToyAirplane",
]

THOR_ZS_OBJECT_TYPES = [
    "Apple",
    "BasketBall",
    "HousePlant",
    "Television",
]

THOR_NON_ZS_OBJECT_TYPES = [
    "AlarmClock",
    "BaseballBat",
    "Bowl",
    "GarbageCan",
    "Laptop",
    "Mug",
    "SprayBottle",
    "Vase",
]

THOR_OBJECT_TYPES_CLIP = [
    "alarm clock",
    "apple",
    "baseball bat",
    "basketball",
    "bowl",
    "garbage can",
    "plant",
    "laptop",
    "mug",
    "spray bottle",
    "TV",
    "vase"
]

THOR_OBJECT_TYPES_MAP = {THOR_OBJECT_TYPES[i]: THOR_OBJECT_TYPES_CLIP[i] for i in range(len(THOR_OBJECT_TYPES))}

# GPT-3.5 generated priors
# where can one usually find {} in a house
# {} are usually
# hparam: temp = 0.0
# hparam: max length = 256
GPT_THOR_OBJECT_TYPES_CLIP = [
    'alarm clock in the bedroom or living room',
    'apple in the kitchen or dining room',
    'baseball bat in a garage, basement, or bedroom',
    'basketball in a garage, basement, or bedroom',
    'bowl in the kitchen, living room, or dining room',
    'garbage can in the kitchen, bathrooms, and other common areas of a house',
    'plant in a living room, bedroom, or kitchen',
    'laptop in a living room, bedroom, or home office',
    'mug in the kitchen, living room, or bedroom',
    'spray bottle in the kitchen, bathroom, or laundry room',
    'TV in the living room, bedroom, or family room',
    'vase in the living room, bedroom, or dining room'
]

THOR_PRIORS = [
    'bedroom or living room',
    'kitchen or dining room',
    'garage, basement, or bedroom',
    'garage, basement, or bedroom',
    'kitchen, living room, or dining room',
    'kitchen, bathrooms, and other common areas of a house',
    'living room, bedroom, or kitchen',
    'living room, bedroom, or home office',
    'kitchen, living room, or bedroom',
    'kitchen, bathroom, or laundry room',
    'living room, bedroom, or family room',
    'living room, bedroom, or dining room'
]

GPT_LONGTAIL_THOR_OBJECT_TYPES_CLIP = [
    'gingerbread house in the kitchen or dining room',
    'espresso machine in a kitchen or dining room',
    'green plastic crate in a garage, basement, or storage room',
    'white electric guitar in a bedroom or living room',
    'rice cooker in the kitchen',
    'llama wicker basket in a living room or bedroom',
    'whiteboard saying cvpr in a home office or study',
    "tie dye surfboard in a beach house's living room or bedroom",
    "blue and red tricycle in a child's bedroom or playroom",
    'graphics card in a computer room or a gaming room',
    'mate gourd in a kitchen or living room',
    "wooden toy plane in a child's bedroom or playroom"
]

LONGTAIL_PRIORS = [
    'kitchen or dining room',
    'kitchen or dining room',
    'garage, basement, or storage room',
    'bedroom or living room',
    'kitchen',
    'living room or bedroom',
    'home office or study',
    "beach house's living room or bedroom",
    "child's bedroom or playroom",
    'computer room or a gaming room',
    'kitchen or living room',
    "child's bedroom or playroom"
]

HABITAT_OBJECT_TYPES = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]

THOR_LONGTAIL_OBJECT_TYPES_CLIP = [
    "gingerbread house",
    "espresso machine",
    "green plastic crate",
    "white electric guitar",
    "rice cooker",
    "llama wicker basket",
    "whiteboard saying cvpr",
    "tie dye surfboard",
    "blue and red tricycle",
    "graphics card",
    "mate gourd",
    "wooden toy plane",
]

HABITAT_OBJECT_TYPES_CLIP = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "dresser",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "TV",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]

GPT_HABITAT_OBJECT_TYPES_CLIP = [
    'chair in the living room, bedroom, dining room, or office',
    'table in the living room, dining room, kitchen, or family room',
    'picture in a living room, bedroom, hallway, or dining room',
    'cabinet in a kitchen, bathroom, bedroom, or living room',
    'cushion in the living room, bedroom, or den',
    'sofa in the living room, family room, or den',
    'bed in the bedroom, guest room, or master suite of a house',
    'dresser in a bedroom, living room, or hallway',
    'plant in a living room, bedroom, or kitchen',
    'sink in the kitchen, bathroom, laundry room, and/or utility room',
    'toilet in a bathroom or powder room',
    'stool in a kitchen, living room, or bedroom',
    'towel in a bathroom, kitchen, or laundry room',
    'TV in the living room, bedroom, or family room',
    'shower in a bathroom', 'bathtub in a bathroom or a master suite',
    'counter in a kitchen, bathroom, or laundry room',
    'fireplace in the living room, family room, or den of a house',
    'gym equipment in a home gym or exercise room',
    'seating in the living room, family room, or den',
    'clothes in a bedroom, closet, or laundry room'
]

HABITAT_PRIORS = [
    'living room, bedroom, dining room, or office',
    'living room, dining room, kitchen, or family room',
    'living room, bedroom, hallway, or dining room',
    'kitchen, bathroom, bedroom, or living room',
    'living room, bedroom, or den',
    'living room, family room, or den',
    'bedroom, guest room, or master suite of a house',
    'bedroom, living room, or hallway',
    'living room, bedroom, or kitchen',
    'kitchen, bathroom, laundry room, and/or utility room',
    'bathroom or powder room',
    'kitchen, living room, or bedroom',
    'bathroom, kitchen, or laundry room',
    'living room, bedroom, or family room',
    'bathroom', 'bathtub in a bathroom or a master suite',
    'kitchen, bathroom, or laundry room',
    'living room, family room, or den of a house',
    'home gym or exercise room',
    'living room, family room, or den',
    'bedroom, closet, or laundry room'
]

# where can one usually find {} in a house

# {} are usually

# temp = 0.0
# max length = 256
GPT_HABITAT_OBJECT_TYPES_CLIP = [
    'chair in the living room, bedroom, or office',
    'table in the living room, dining room, or kitchen',
    'picture in a house in the living room, bedroom, or kitchen',
    'cabinet in the kitchen, living room, or bedroom',
    'cushion on a sofa in the living room',
    'sofa in the living room, family room, or den',
    'bed in a house in the bedroom, guest room, or in some cases, the living room',
    'dresser in a house in the bedroom, living room, or dining room',
    'plant in a house in the living room, on a windowsill',
    'sink in the bathroom, kitchen, or laundry room',
    'toilet in a house in the bathroom',
    'stool in the kitchen or in the bathroom',
    'towel in a house in the bathroom or near a pool',
    'TV in a house in the living room, family room, or den',
    'shower in a house in the bathroom',
    'bathtub in a house in the bathroom',
    'counter in the kitchen or bathroom',
    'fireplace in a house in the living room, family room, or den',
    'gym equipment in a house in the basement or in a room that has been converted to a gym',
    'seating in a house in the living room, den, or family room',
    'clothes in a house in the bedroom or in the closet'
]

OBJECT_TYPES_WITH_PROPERTIES = {
    "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
    "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
    "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
    "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
    "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
    "Window": {"openable": False, "receptacle": False, "pickupable": False},
    "Sink": {"openable": False, "receptacle": True, "pickupable": False},
    "Floor": {"openable": False, "receptacle": True, "pickupable": False},
    "Book": {"openable": True, "receptacle": False, "pickupable": True},
    "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Knife": {"openable": False, "receptacle": False, "pickupable": True},
    "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
    "Bread": {"openable": False, "receptacle": False, "pickupable": True},
    "Fork": {"openable": False, "receptacle": False, "pickupable": True},
    "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
    "Potato": {"openable": False, "receptacle": False, "pickupable": True},
    "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
    "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
    "Pan": {"openable": False, "receptacle": True, "pickupable": True},
    "Plate": {"openable": False, "receptacle": True, "pickupable": True},
    "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
    "Vase": {"openable": False, "receptacle": False, "pickupable": True},
    "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
    "Egg": {"openable": False, "receptacle": False, "pickupable": True},
    "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
    "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Pot": {"openable": False, "receptacle": True, "pickupable": True},
    "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
    "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
    "Cup": {"openable": False, "receptacle": True, "pickupable": True},
    "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
    "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
    "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
    "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
    "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
    "Apple": {"openable": False, "receptacle": False, "pickupable": True},
    "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
    "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
    "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
    "Mug": {"openable": False, "receptacle": True, "pickupable": True},
    "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
    "Statue": {"openable": False, "receptacle": False, "pickupable": True},
    "Stool": {"openable": False, "receptacle": True, "pickupable": False},
    "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
    "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
    "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
    "Chair": {"openable": False, "receptacle": True, "pickupable": False},
    "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
    "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
    "Pen": {"openable": False, "receptacle": False, "pickupable": True},
    "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
    "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
    "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
    "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
    "Safe": {"openable": True, "receptacle": True, "pickupable": False},
    "Painting": {"openable": False, "receptacle": False, "pickupable": False},
    "Box": {"openable": True, "receptacle": True, "pickupable": True},
    "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
    "Television": {"openable": False, "receptacle": False, "pickupable": False},
    "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
    "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
    "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
    "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
    "Watch": {"openable": False, "receptacle": False, "pickupable": True},
    "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
    "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
    "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
    "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
    "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
    "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
    "Boots": {"openable": False, "receptacle": False, "pickupable": True},
    "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
    "Desk": {"openable": False, "receptacle": True, "pickupable": False},
    "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
    "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
    "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
    "Candle": {"openable": False, "receptacle": False, "pickupable": True},
    "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
    "Bed": {"openable": False, "receptacle": True, "pickupable": False},
    "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
    "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
    "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
    "CD": {"openable": False, "receptacle": False, "pickupable": True},
    "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
    "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
    "Poster": {"openable": False, "receptacle": False, "pickupable": False},
    "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
    "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
    "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
    "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
    "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
    "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
    "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
    "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
    "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
    "Towel": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
    "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
    "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
    "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
    "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
    "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
    "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
    "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
    "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
}
