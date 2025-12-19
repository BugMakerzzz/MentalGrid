ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

At the initial state, you are located at the camera position.
The camera's initial facing direction defines North (a fixed global direction).
Forward/Right/Left/Behind is defined relative to your current facing direction and may change over time.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.
2. **Actions (Optional)** - a natural language sequence of actions describing the movements of you (i.e., the camera) and the objects. Always treat the direction the current camera is facing as the forward direction. We ignore any blockages while moving.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - 5 answer candidates, each tagged by a capital letter (A, B, C, D, E).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
The last line of your response should be of the following format: # Answer: A/B/C/D/E
"""

DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

At the initial state, you are located at the camera position.
The camera’s initial facing direction defines North (a fixed global direction).
Forward/Right/Left/Behind is defined relative to your current facing direction and may change over time.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Actions (Optional)** - a natural language sequence of actions describing the movements of you (i.e., the camera) and the objects. Always treat the direction the current camera is facing as the forward direction. We ignore any blockages while moving.
3. **Question** - a natural-language query about spatial relationships between objects in the image.  
4. **Options** - 5 answer candidates, each tagged by a capital letter (A, B, C, D, E).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
The last line of your response should be of the following format: # Answer: A/B/C/D
"""

MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Actions (Optional)** - a natural language sequence of actions describing the movements of you (i.e., the camera) and the objects. Always treat the direction the current camera is facing as the forward direction. We ignore any blockages while moving.
3. **Question** - a natural-language query about spatial relationships between objects in the image.  
4. **Options** - 4 answer candidates, each tagged by a capital letter (A, B, C, D).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.
4. Finally, select the most appropriate option (A, B, C or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
"""

GET_OBJECTS_SYSTEM_PROMPT  = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Question** - a natural-language query about spatial relationships between objects.  
2. **Options** - 4 answer candidates, each tagged by a capital letter (A, B, C, D).

Your task is to provide all objects that are included in question and options.
Always ground your answer in the question and options, do not hallucinate unseen objects.
Your response should be of the following format: "# Answer: <object 1>. <object 2>. ... <object n>"
Do not add any extra thought, just output the response in the form mentioned above.

# Example 1
In which direction is the indoor_plant located relative to the table?\n(A) Left Forward (B) Left Backward (C) Right Forward (D) Forward
# Answer: indoor_plant. table

# Examples 2
Suppose we perform the following movements in the image: First, the table moves forward by 1.5 meters. Finally, the chair moves backward by 0.5 meters. If you are in the camera's viewpoint, which object is the closest to you?\n(A) chair (B) wall_cabinet (C) tv_stand (D) table
# Answer: table. chair. camera. wall_cabinet. tv_stand
"""

MAP_COT_SYSTEM_PROMPT =  """
You are a spatial-reasoning assistant.

At the initial state, you are located at the camera position.
The camera's initial facing direction defines North (a fixed global direction).
Forward/Right/Left/Behind is defined relative to your current facing direction and may change over time.

Task
-----
You will receive  
1. **Maps** - A JSON format text input, indicating the coordinates corresponding to each object.
2. **Question** - a natural-language query about spatial relationships between objects in the map.
3. **Options** - 5 answer candidates, each tagged by a capital letter (A, B, C, D, E).

Think step by step and provide the answer based on the map.
Always ground your answer in the map; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
The last line of your response should be of the following format: # Answer: A/B/C/D/E
"""

