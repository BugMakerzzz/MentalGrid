from enum import Enum

# ================================================================
# 1. Query Types
# ================================================================
class TaskType(Enum):
    SSP = "static_spatial_perception"
    FSP = "final_state_prediction"
    DPT = "dynamic_process_tracking"


class QueryType(Enum):
    # distance
    OBJ_DIS = "object_to_distance"
    DIS_OBJ = "distance_to_object"

    # direction
    OBJ_DIR = "object_to_direction"
    DIR_OBJ = "direction_to_object"



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


EGO_DIRECTIONS = [
    "forward", "right-forward", "right", "right-backward",
    "backward", "left-backward", "left", "left-forward"
]

ALLO_DIRECTIONS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest"
]

ALL_SCENES = [
    'ArchitecTHOR', 'iTHOR', 'ProcTHOR', 'replica', 'RoboTHOR'
]


QUESTION_TEMPLATES = {
    QueryType.OBJ_DIS: {
        TaskType.SSP: {
            Centric.EGO: [
                "How far are you from the {target_obj} in meters?",
                
                "Provide an approximate distance from your position to the {target_obj} (m).",
                
                "Roughly how many meters away is the {target_obj} from you?",
                
                "Give a numerical meter estimate of your distance to the {target_obj}.",
                
                "Please estimate the distance between you and the {target_obj} (in meters)."
            ],
            Centric.ALLO: [
                "How far is the {target_obj} from the {view_obj} (in meters)?",
                
                "Approximately how many meters apart are the {view_obj} and the {target_obj}?",
                
                "Report the distance between the {view_obj} and the {target_obj} in meters.",
                
                "Provide a numerical estimate (m) of the distance from {view_obj} to {target_obj}.",
                
                "Please estimate the distance between the {target_obj} and the {view_obj} (in meters)."
            ]
        },
        
        TaskType.FSP: {
            Centric.EGO: [
                "{action_text}\nWhat is the final distance (meters) between you and the {target_obj}?",
                
                "{action_text}\nProvide an approximate final distance from your position to the {target_obj} (m).",
                
                "{action_text}\nAfter the movements, roughly how many meters away is the {target_obj} from you?",
                
                "{action_text}\nGive a numerical meter estimate of your final distance to the {target_obj}.",

                "{action_text}\nPlease estimate the final distance between you and the {target_obj} (in meters)."
            ],
            
            Centric.ALLO: [
                "{action_text}\nWhat is the final distance (meters) between the {target_obj} and the {view_obj}?",
                
                "{action_text}\nAfter the movements, approximately how many meters apart are the {view_obj} and the {target_obj}?",
                
                "{action_text}\nReport the final distance between the {view_obj} and the {target_obj} in meters.",
                
                "{action_text}\nProvide a numerical estimate (m) of the final distance from {view_obj} to {target_obj}."
                
                "{action_text}\nPlease estimate the final distance between the {target_obj} and the {view_obj} (in meters)."
            ]
        },
        
        TaskType.DPT: {
            Centric.EGO: [
                "{action_text}\nDuring the movement sequence, the per-step change in distance (meters) between you and {target_obj} is:\n"
                "{masked_seq}\nPlease fill in the correct values for each [MASK] token.",
                
                "{action_text}\nThroughout the motion, the step-wise meter change in separation between you and {target_obj} is:\n"
                "{masked_seq}\nInsert the appropriate numbers into the [MASK] tokens.",
                
                "{action_text}\nOver the course of the movement, the distance variation (meters) between you and {target_obj} at each step is:\n"
                "{masked_seq}\nProvide the correct values for all [MASK] placeholders.",
                
                "{action_text}\nDuring the dynamic process, the incremental change in distance (m) between you and {target_obj} per step is:\n"
                "{masked_seq}\nFill each [MASK] slot with the proper value.",
             
                "{action_text}\nDuring the movement process, the distance changes (in meters) between you and {target_obj} at each step are:\n"
                "{masked_seq}\nPlease fill in the correct values for each [MASK] token."
            ],
            Centric.ALLO: [
                "{action_text}\nDuring the movement sequence, the per-step change in distance (meters) between {target_obj} and {view_obj} is:\n"
                "{masked_seq}\nPlease fill in the correct values for each [MASK] token.",
                
                "{action_text}\nThroughout the motion, the step-wise meter change in separation between {target_obj} and {view_obj} is:\n"
                "{masked_seq}\nInsert the appropriate numbers into the [MASK] tokens.",
                
                "{action_text}\nOver the course of the movement, the distance variation (meters) between {view_obj} and {target_obj} at each step is:\n"
                "{masked_seq}\nProvide the correct values for all [MASK] placeholders.",
                
                "{action_text}\nDuring the dynamic process, the incremental change in distance (m) between {view_obj} and {target_obj} per step is:\n"
                "{masked_seq}\nFill each [MASK] slot with the proper value.",
            
                "{action_text}\nDuring the movement process, the distance changes (in meters) between {target_obj} and {view_obj} at each step are:\n"
                "{masked_seq}\nPlease fill in the correct values for each [MASK] token."
            ]
        }
    },
    
    QueryType.DIS_OBJ: {
        TaskType.SSP: {
            Centric.EGO: [
                "Which object lies farthest from your position?\n{options}",
                "Among these options, which object is the most distant from you?\n{options}",
                "Which of the listed objects is located the farthest away from you?\n{options}",
                "Which object has the greatest distance from you?\n{options}",
                "Select the object that is farthest from you.\n{options}",

                "Which object lies closest to your position?\n{options}",
                "Among these options, which object is nearest to you?\n{options}",
                "Which of the listed objects is located the closest to you?\n{options}",
                "Which object has the smallest distance from you?\n{options}",
                "Select the object that is closest to you.\n{options}"
            ],
        
            Centric.ALLO: [
                "Which object lies farthest from the {target_obj}?\n{options}",
                "Among these options, which object is the most distant from the {target_obj}?\n{options}",
                "Which of the listed objects is located the farthest away from the {target_obj}?\n{options}",
                "Which object has the greatest distance from the {target_obj}?\n{options}",
                "Select the object that is farthest from the {target_obj}.\n{options}",

                "Which object lies closest to the {target_obj}?\n{options}",
                "Among these options, which object is nearest to the {target_obj}?\n{options}",
                "Which of the listed objects is located the closest to the {target_obj}?\n{options}",
                "Which object has the smallest distance from the {target_obj}?\n{options}",
                "Select the object that is closest to the {target_obj}.\n{options}"
            ]
        },
        
        TaskType.FSP: {
            Centric.EGO: [
                "{action_text}\nAfter the movements, which object ends up farthest from you?\n{options}",
                "{action_text}\nAfter performing the actions, which object is the most distant from your position?\n{options}",
                "{action_text}\nFollowing the movements, which of the listed objects is located the farthest away from you?\n{options}",
                "{action_text}\nAfter the sequence, which object has the greatest final distance from you?\n{options}",
                "{action_text}\nAfter the actions, select the object that ends up the farthest from you.\n{options}",

                "{action_text}\nAfter the movements, which object ends up closest to you?\n{options}",
                "{action_text}\nAfter performing the actions, which object is nearest to your position?\n{options}",
                "{action_text}\nFollowing the movements, which of the listed objects is located the closest to you?\n{options}",
                "{action_text}\nAfter the sequence, which object has the smallest final distance from you?\n{options}",
                "{action_text}\nAfter the actions, select the object that ends up the closest to you.\n{options}"
            ],
        
            Centric.ALLO: [
                "{action_text}\nAfter the movements, which object ends up farthest from the {target_obj}?\n{options}",
                "{action_text}\nAfter performing the actions, which object is the most distant from the {target_obj}?\n{options}",
                "{action_text}\nFollowing the movements, which of the listed objects is located the farthest from the {target_obj}?\n{options}",
                "{action_text}\nAfter the sequence, which object has the greatest final distance from the {target_obj}?\n{options}",
                "{action_text}\nAfter the actions, select the object that ends up the farthest from the {target_obj}.\n{options}",

                "{action_text}\nAfter the movements, which object ends up closest to the {target_obj}?\n{options}",
                "{action_text}\nAfter performing the actions, which object is nearest to the {target_obj}?\n{options}",
                "{action_text}\nFollowing the movements, which of the listed objects is located the closest to the {target_obj}?\n{options}",
                "{action_text}\nAfter the sequence, which object has the smallest final distance from the {target_obj}?\n{options}",
                "{action_text}\nAfter the actions, select the object that ends up the closest to the {target_obj}.\n{options}"
            ]
        },
        
        TaskType.DPT: {
            Centric.EGO: [
                "{action_text}\nDuring the movement, the objects furthest from you, in sequence, are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAs the movement progresses, the farthest object from you evolves in the following order:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nThroughout the movement, the farthest object from you changes over time as:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nWhile moving, the farthest object relative to you transitions in the sequence:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the dominant farthest object from you at each phase appears in the following sequence:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}",


                "{action_text}\nDuring the movement, the objects closest to you, in sequence, are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAs the movement progresses, the closest object to you evolves in the following order:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nThroughout the movement, the closest object to you changes over time as:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nWhile moving, the closest object relative to you transitions in the sequence:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the dominant closest object to you appears in the following sequence:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}"
            ],
            Centric.ALLO: [
                "{action_text}\nDuring the movement, the objects furthest from the {target_obj}, in sequence, are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAs the movement progresses, the farthest object from the {target_obj} evolves in the following order:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nThroughout the movement, the farthest object from the {target_obj} changes over time as:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nWhile moving, the farthest object relative to the {target_obj} transitions in the sequence:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the dominant farthest object from the {target_obj} appears in the following sequence:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}",


                "{action_text}\nDuring the movement, the objects closest to the {target_obj}, in sequence, are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAs the movement progresses, the closest object to the {target_obj} evolves in the following order:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nThroughout the movement, the closest object to the {target_obj} changes over time as:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nWhile moving, the closest object relative to the {target_obj} transitions in the sequence:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the dominant closest object to the {target_obj} appears in the following sequence:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}"
            ]
        }
    },


    QueryType.OBJ_DIR: {
        TaskType.SSP: {
            Centric.EGO: [
                "In which direction is the {target_obj} positioned relative to you?\n{options}",
                "From your viewpoint, where is the {target_obj} located in terms of direction?\n{options}",
                "Relative to your position, what direction does the {target_obj} lie?\n{options}",
                "From your perspective, in which direction does the {target_obj} lie relative to you?\n{options}",
                "Which directional bearing does the {target_obj} have relative to you?\n{options}",

                "From your perspective, in which direction is the {target_obj} located relative to the {view_obj}?\n{options}",
                "Relative to the {view_obj}, where is the {target_obj} positioned from your viewpoint?\n{options}",
                "From your standpoint, in which direction does the {target_obj} lie with respect to the {view_obj}?\n{options}",
                "According to your perspective, which direction does the {target_obj} occupy relative to the {view_obj}?\n{options}",
                "From where you are, which direction does the {target_obj} lie relative to the {view_obj}?\n{options}"
            ],
            Centric.ALLO: [
                "In which direction is the {target_obj} located relative to the {view_obj}?\n{options}",
                "Relative to the {view_obj}, where is the {target_obj} positioned directionally?\n{options}",
                "From the perspective of the {view_obj}, which direction does the {target_obj} lie?\n{options}",
                "In directional terms, where does the {target_obj} lie relative to the {view_obj}?\n{options}",
                "Which direction does the {target_obj} occupy relative to the {view_obj}?\n{options}"
            ]
        },
        
        TaskType.FSP: {
            Centric.EGO: [
                "{action_text}\nAfter the movements, in which direction does the {target_obj} end up relative to you?\n{options}",
                "{action_text}\nAfter completing the actions, where does the {target_obj} end up from your viewpoint in directional terms?\n{options}",
                "{action_text}\nFollowing the movement, what direction does the {target_obj} lie relative to your final position?\n{options}",
                "{action_text}\nAfter the sequence, in which direction does the {target_obj} lie relative to you?\n{options}",
                "{action_text}\nAfter the actions conclude, which directional bearing does the {target_obj} have relative to you?\n{options}",

                "{action_text}\nAfter the movements, in which direction does the {target_obj} end up relative to the {view_obj} from your perspective?\n{options}",
                "{action_text}\nAfter completing the actions, where is the {target_obj} positioned relative to the {view_obj} from your viewpoint?\n{options}",
                "{action_text}\nFollowing the movement, what direction does the {target_obj} lie with respect to the {view_obj} from your standpoint?\n{options}",
                "{action_text}\nAfter the sequence, which direction does the {target_obj} occupy relative to the {view_obj} from your perspective?\n{options}",
                "{action_text}\nAfter the actions conclude, which direction does the {target_obj} lie relative to the {view_obj} from where you are?\n{options}"
            ],

            Centric.ALLO: [
                "{action_text}\nAfter the movements, in which direction does the {target_obj} end up relative to the {view_obj}?\n{options}",
                "{action_text}\nAfter completing the actions, where is the {target_obj} positioned directionally relative to the {view_obj}?\n{options}",
                "{action_text}\nFollowing the movement, which direction does the {target_obj} lie from the perspective of the {view_obj}?\n{options}",
                "{action_text}\nAfter the sequence, where does the {target_obj} lie relative to the {view_obj} in directional terms?\n{options}",
                "{action_text}\nAfter the actions conclude, which direction does the {target_obj} occupy relative to the {view_obj}?\n{options}"
            ]
        },
        
        TaskType.DPT: {
            Centric.EGO: [
                "{action_text}\nDuring the movement process, the changes in the relative direction angle from you to the {target_obj} at each step are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAcross the movement process, the relative bearing angle from you to the {target_obj} varies as follows:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nWhile moving, the angle of the relative direction from you to the {target_obj} changes in the sequence:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nThroughout the motion, the relative directional angle from you toward the {target_obj} evolves step by step as:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the change in relative direction angle from you to the {target_obj} at each stage is:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}",


                "{action_text}\nDuring the movement process, from your perspective, the changes in the relative direction angle from the {view_obj} to the {target_obj} at each step are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAcross the movement process, from your viewpoint, the relative bearing angle from the {view_obj} to the {target_obj} varies as follows:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nWhile moving, the angle of the relative direction from the {view_obj} to the {target_obj} changes in the sequence from your perspective:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nThroughout the motion, the relative directional angle from the {view_obj} toward the {target_obj} evolves step by step as seen from your view:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, from your standpoint, the change in relative direction angle from the {view_obj} to the {target_obj} at each stage is:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}"
            ],
            
            Centric.ALLO:  [
                "{action_text}\nDuring the movement process, the changes in the relative direction angle from the {view_obj} to the {target_obj} at each step are:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nAcross the movement process, the relative bearing angle from the {view_obj} to the {target_obj} varies as follows:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nWhile moving, the angle of the relative direction from the {view_obj} to the {target_obj} changes in the sequence:\n"
                "{masked_seq}\nPick the option that correctly fills the [MASK] placeholders.\n{options}",

                "{action_text}\nThroughout the motion, the relative directional angle from the {view_obj} toward the {target_obj} evolves step by step as:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nDuring motion, the change in relative direction angle from the {view_obj} to the {target_obj} at each stage is:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}"
            ]
        }
    },
    QueryType.DIR_OBJ: {
        TaskType.SSP: {
            Centric.EGO: [
                "Which object lies in your {direction}?\n{options}",
                "From where you stand, which object is positioned in your {direction}?\n{options}",
                "Relative to you, which object is located in the {direction} direction?\n{options}",
                "According to your viewpoint, which object falls within your {direction}?\n{options}",
                "In your frame of reference, which object appears in your {direction}?\n{options}",

                "From your perspective, which object is positioned in the {direction} of the {target_obj}?\n{options}",
                "Relative to the {target_obj}, which object lies in the {direction} from your viewpoint?\n{options}",
                "From where you are, which object is located in the {direction} with respect to the {target_obj}?\n{options}",
                "According to your viewpoint, which object appears in the {direction} of the {target_obj}?\n{options}",
                "In your frame of reference, which object occupies the {direction} of the {target_obj}?\n{options}"
            ],
            Centric.ALLO: [
                "Which object is located in the {direction} of the {target_obj}?\n{options}",
                "Relative to the {target_obj}, which object lies in the {direction}?\n{options}",
                "In directional terms, which object falls in the {direction} of the {target_obj}?\n{options}",
                "From the standpoint of the {target_obj}, which object occupies the {direction} region?\n{options}",
                "Which object is positioned in the {direction} with respect to the {target_obj}?\n{options}"
            ]
        },
        
        TaskType.FSP: {
            Centric.EGO: [
                "{action_text}\nAfter the movements, which object ends up in your {direction}?\n{options}",
                "{action_text}\nAfter completing the actions, which object is positioned in your {direction} from your viewpoint?\n{options}",
                "{action_text}\nFollowing the movement, which object lies in the {direction} relative to you?\n{options}",
                "{action_text}\nAfter the sequence, which object appears in your {direction}?\n{options}",
                "{action_text}\nOnce the actions conclude, which object occupies your {direction}?\n{options}",

                "{action_text}\nAfter the movements, which object ends up in the {direction} of the {target_obj} from your perspective?\n{options}",
                "{action_text}\nAfter completing the actions, which object is located in the {direction} with respect to the {target_obj} from your viewpoint?\n{options}",
                "{action_text}\nFollowing the movement, which object lies in the {direction} of the {target_obj} as seen by you?\n{options}",
                "{action_text}\nAfter the sequence, which object appears in the {direction} of the {target_obj} from where you are?\n{options}",
                "{action_text}\nOnce the actions conclude, which object occupies the {direction} region with respect to the {target_obj} from your perspective?\n{options}"
            ],
            Centric.ALLO: [
                "{action_text}\nAfter the movements, which object ends up in the {direction} of the {target_obj}?\n{options}",
                "{action_text}\nAfter completing the actions, which object is positioned in the {direction} with respect to the {target_obj}?\n{options}",
                "{action_text}\nFollowing the movement, which object lies in the {direction} relative to the {target_obj}?\n{options}",
                "{action_text}\nAfter the sequence, which object appears in the {direction} of the {target_obj}?\n{options}",
                "{action_text}\nOnce the actions conclude, which object occupies the {direction} region with respect to the {target_obj}?\n{options}"
            ]
        },
        
        TaskType.DPT: {
            Centric.EGO: [
            "{action_text}\nOver the course of the movement, the object that falls within your {direction}, in sequence, is:\n"
            "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

            "{action_text}\nThroughout the movement, the object appearing in your {direction}, in changing order, is:\n"
            "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

            "{action_text}\nDuring motion, the object that remains in your {direction}, in ordered segments, is:\n"
            "{masked_seq}\nPick the option that correctly fills in the [MASK] placeholders.\n{options}",

            "{action_text}\nAcross the movement, the object occupying your {direction}, as the dominant one over time, is:\n"
            "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

            "{action_text}\nWhile moving, the object located in your {direction}, in sequential segments, is:\n"
            "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}",

            "{action_text}\nDuring the movement, from your perspective, the object located in the {direction} of the {target_obj}, in sequence, is:\n"
            "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

            "{action_text}\nThroughout the motion, from your viewpoint, the object occupying the {direction} of the {target_obj}, in ordered segments, is:\n"
            "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

            "{action_text}\nDuring motion, as seen from your position, the object that appears in the {direction} of the {target_obj}, in changing order, is:\n"
            "{masked_seq}\nPick the option that correctly fills in the [MASK] placeholders.\n{options}",

            "{action_text}\nAcross the sequence, from your standpoint, the object falling in the {direction} of the {target_obj}, as the dominant one, is:\n"
            "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

            "{action_text}\nWhile moving, according to your viewpoint, the object located in the {direction} of the {target_obj}, in sequence, is:\n"
            "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}",
        ],
            
            Centric.ALLO: [
                "{action_text}\nDuring the movement process, the object occupying the {direction} of the {target_obj}, in sequence, is:\n"
                "{masked_seq}\nPlease choose the option that correctly fills in the [MASK] tokens.\n{options}",

                "{action_text}\nThroughout the movement, the dominant object in the {direction} of the {target_obj}, in order of appearance, is:\n"
                "{masked_seq}\nSelect the answer that completes the [MASK] tokens.\n{options}",

                "{action_text}\nAcross the movement, the object visible in the {direction} of the {target_obj}, segment by segment, is:\n"
                "{masked_seq}\nPick the option that correctly fills in the [MASK] placeholders.\n{options}",

                "{action_text}\nDuring motion, the sequence of objects occupying the {direction} of the {target_obj} is:\n"
                "{masked_seq}\nChoose the correct option for the [MASK] entries.\n{options}",

                "{action_text}\nAs the movement unfolds, the object in the {direction} of the {target_obj}, in changing order, is:\n"
                "{masked_seq}\nSelect the option that completes all [MASK] slots.\n{options}"
            ]
        }
    }
}

ACTION_PREFIX = [
     "Given the following movement sequence from the reference view: ",
    "Starting from the reference view and performing: ",
    "After executing these actions from the reference view: ",
    "From the reference view, suppose the following movements occur: ",
    "Consider the following movements beginning at the reference view: " 
]