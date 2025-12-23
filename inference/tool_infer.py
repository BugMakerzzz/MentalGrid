"""
Main pipeline for tool use
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from tools.depth import DepthModule
from tools.segmentation import SegmentationModule
# from tools.render import RenderModule
from eval.load_data import add_message
from .model import Model
from utils.prompt import GET_OBJECTS_SYSTEM_PROMPT, MAP_COT_SYSTEM_PROMPT

import numpy as np


class ToolPipeline:
    def __init__(
        self,
        config,
        device: str = 'cuda',
    ):
        self.config = config
        self.device = device

        # Initialize VLM model
        self.model = Model(config=config.vlm)
        print(f"Initialized VLMs")

        # Initialize vision modules
        self.segmentation_module = SegmentationModule(config.segmentation, device)
        self.depth_module = DepthModule(config.depth, device)
        print(f"Initialized vision modules")

        # Load renderer
        # self.render_module = RenderModule(device=device)
        
        # Set color dictionary (for visual prompts)
        self.color_dict = [
            ['blue', [0, 0, 1]],
            ['red', [1, 0, 0]],
            ['green', [0, 1, 0]],
            ['yellow', [1, 1, 0]],
            ['purple', [1, 0, 1]],
            ['orange', [1, 0.5, 0]],
            ['brown', [0.5, 0.25, 0]],
            ['gray', [0.5, 0.5, 0.5]],
        ]


    def get_objects(
        self,
        prompt: str,
        num_tries: int = 2,
    ) -> Tuple[Optional[List[str]], Optional[list]]:
        
        response_objs = None
        # prompt_objs = None

        for try_idx in range(num_tries):
            
            messages = add_message(
                [],
                role="system",
                text=GET_OBJECTS_SYSTEM_PROMPT
            )
            
            messages = add_message(
                messages,
                role="user",
                text= prompt,
            )
            response_objs = self.model.generate(messages)

            match_objs = response_objs.strip().split('.')

            if len(match_objs) > 0:
                break
        
        objs_of_interest = response_objs.strip()

        return objs_of_interest


    def generate(
        self,
        example,
        # perspective_prompt_type: str = "visual",
        return_conv_history: bool = True,
    ):
       
        conv_history = {} if return_conv_history else None

        # [1] Scene Abstraction
        objects_str = self.get_objects(prompt=example['query'])
        conv_history['detect_obj'] = objects_str


        seg_out = self.segmentation_module.run_segmentation(image_path=example['image_path'],
                                                                                  prompt=objects_str)

        object_locations = self.depth_module.run_depth(image_path=example['image_path'], seg_out=seg_out)
        conv_history['obj_location'] =  object_locations
        
        messages = add_message(
                [],
                role="system",
                text=MAP_COT_SYSTEM_PROMPT
            )
            
        input = f"Map: {object_locations}\nQuestion: {example['query']}"
        messages = add_message(
            messages,
            role="user",
            text=input,
        )
        response = self.model.generate(messages)
        conv_history['final_response'] = response
        # abstract_scene_dict[ref_viewer] = do_perspective_change(ctx, abstract_scene_dict["camera"], ref_viewer=ref_viewer)
        # ctx.logger.info(f"[Perspective Change] Abstracted scene ({ref_viewer}): {abstract_scene_dict[ref_viewer]}")

        # # [3] Perspective Prompting
        # if perspective_prompt_type == "visual":
        #     response_apc, conv_history = do_perspective_prompting_visual(
        #         ref_viewer,
        #         abstract_scene_dict[ref_viewer],
        #         prompt,
        #         conv_history=conv_history,
        #     )
        # else:
        #     raise ValueError(f"Invalid perspective prompt type: {perspective_prompt_type}")

        # if conv_history is not None:
        #     conv_history += [
        #         {"text": "[Question]\n\n" + prompt, "image": image.resize((400, 400))},
        #         {"text": "[Response (APC)]\n\n" + response_apc, "image": None},
        #     ]

        return response, conv_history
