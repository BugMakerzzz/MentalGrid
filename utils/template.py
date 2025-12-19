from enum import Enum

# ================================================================
# 1. Query Types
# ================================================================

class QueryType(Enum):
    # distance
    DISTANCE_EVALUATION = "distance_evaluation"
    DISTANCE_LOCATION = "distance_location"
    DISTANCE_TRACKING = "distance_tracking"

    # direction
    DIRECTION_EVALUATION = "direction_evaluation"
    DIRECTION_LOCATION = "direction_location"
    DIRECTION_TRACKING = "direction_tracking"
    
    # both
    DISTANCE_DIRECTION_LOCATION = 'dd_location'
    LOCATION_TRACKING = 'location_tracking'
    MAP_DESCRIPTION = "map_description"


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"


class Centric(Enum):
    EGO = "egocentric"
    ALLO = "allocentric"


class Movement(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    # ALIGN = "align"

EGO_DIRECTIONS = [
    "forward", "right-forward", "right", "right-backward",
    "backward", "left-backward", "left", "left-forward"
]

ALLO_DIRECTIONS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest"
]


# ================================================================
# 2. Movement Action Definitions
# ================================================================

QUESTION_TEMPLATES = {
    QueryType.DISTANCE_EVALUATION: {
        Centric.EGO: [
            "Please estimate the distance between you and the {target_obj}.\n{options}"
        ],
        Centric.ALLO: [
            "Please estimate the distance between the {target_obj} and the {view_obj}.\n{options}"
        ]
    },
    QueryType.DISTANCE_LOCATION: {
        Centric.EGO: [
            "Which object is the farthest from you?\n{options}",
            "Which object is the closest to you?\n{options}"
        ],
        Centric.ALLO: [
            "Which object is the farthest from the {target_obj}?\n{options}",
            "Which object is the closest to the {target_obj}?\n{options}"
        ]
    },
    QueryType.DISTANCE_TRACKING: {
        Centric.EGO: [
            "Please describe the change in distance between {target_obj} and you throughout the entire movement process?\n{options}",
        ],
        Centric.ALLO: [
            "Please describe the change in distance between {target_obj} and {view_obj} throughout the entire movement process?\n{options}",
        ]
    },
    QueryType.DIRECTION_EVALUATION: {
        Centric.EGO: [
            "Which direction is the {target_obj} relative to you?\n{options}"
        ],
        Centric.ALLO: [
            "Which direction is the {target_obj} located relative to the {view_obj}?\n{options}"
        ]
    },
    QueryType.DIRECTION_LOCATION: {
        Centric.EGO: [
            "Which object(s) are located in your {direction}?\n{options}"
        ],
        Centric.ALLO: [
            "Which object(s) are located in the {direction} of the {target_obj}?\n{options}"
        ]
    },
    QueryType.DIRECTION_TRACKING: {
        Centric.EGO: [
            "Please describe the change in relative direction between {target_obj} and you throughout the entire movement process?\n{options}",
        ],
        Centric.ALLO: [
            "Please describe the change in relative direction between {target_obj} and {view_obj} throughout the entire movement process?\n{options}"
        ]
    },
    QueryType.DISTANCE_DIRECTION_LOCATION: {
        Centric.EGO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ],
        Centric.ALLO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ]
    },
    QueryType.LOCATION_TRACKING: {
        Centric.EGO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ],
        Centric.ALLO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ]
    },
    QueryType.MAP_DESCRIPTION: {
        Centric.EGO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ],
        Centric.ALLO: [
            "Describe the path from {view_obj} to {target_obj}. Each single step should be output in the format of move forward/backward/left/right. If there is an obstacle in the way, include an extra step formatted as 'go around OBJECT'\n{options}"
        ]
    }
}
