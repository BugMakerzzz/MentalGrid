from openai import OpenAI
# openai config
OPENAI_BASE_URL = 'https://api.gpt.ge/v1/'
OPENAI_API_KEY = 'sk-EokDN7u8SdOfDyytD0Af957eC16544849551767083EfBb9e'
VLLM_API_KEY = 'ljc'

API_MODEL_MAP_DICT = {
    'gpt-4.1': 'gpt-4.1-2025-04-14', # $0.002/K $0.008/K
    'gpt-5': 'gpt-5-2025-08-07', # $0.00125/K $0.01/K
    'gpt-5-mini': 'gpt-5-mini-2025-08-07', # $0.00025/K $0.002/K
    'gpt-5.1': 'gpt-5.1',
    'gpt-4o': 'gpt-4o-2024-11-20', # $0.0025/K $0.01/K
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'o1': 'o1-2024-12-17', # 0.015
    'o1-mini': 'o1-mini-2024-09-12', # 0.003
    'o4-mini': 'o4-mini-2025-04-16', # $0.0011/K  $0.0044/K
    'claude-sonnet-4': 'claude-sonnet-4-20250514', # $0.009/K  $0.045/K
    'claude-sonnet-4-thinking': 'claude-sonnet-4-20250514-thinking',
    'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    'claude-3.7-sonnet-thinking': 'claude-3-7-sonnet-20250219-thinking',# $0.0084/K, $0.042/K
    'gemini-2.5-flash': 'gemini-2.5-flash', # $0.0003/K  $0.0025/K
    'gemini-2.5-pro': 'gemini-2.5-pro', # $0.001/K $0.008/K
    'gemini-3-flash': 'gemini-3-flash-preview',
    'doubao-1.5-vision-pro': 'doubao-1-5-vision-pro-32k',
    'grok-3-reasoning': 'grok-3-reasoning', #0.004
    'gpt-5-mini-high': 'gpt-5-mini-high',
    'gpt-5-mini-minimal': 'gpt-5-mini-minimal',
    'gpt-5-high': 'gpt-5-high',
    'gpt-5-minimal': 'gpt-5-minimal',
    'gpt-4.1-mini': 'gpt-4.1-mini-2025-04-14',
    'gemini-2.5-flash-thinking': 'gemini-2.5-flash-thinking',
    'gemini-2.5-flash-nothinking': 'gemini-2.5-flash-nothinking',
    'doubao-1.5-vision': 'doubao-1-5-vision-pro-32k'
}

VLLM_URL_DICT = {
    120: 'http://175.102.130.120:{port}/v1',
    139: 'http://210.75.240.139:{port}/v1',
    153: 'http://210.75.240.153:{port}/v1',
    154: 'http://210.75.240.154:{port}/v1',
    155: 'http://210.75.240.155:{port}/v1',
    156: 'http://210.75.240.156:{port}/v1'
}


class Model():

    def __init__(self, config):
        self.config = config
        model_name = config.model_name
        self.api_model = True if model_name in API_MODEL_MAP_DICT.keys() else False
        if self.api_model:
            self.client = OpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                default_headers={"x-foo": "true"}
            )
            self.model_path = API_MODEL_MAP_DICT[model_name]
        else:    
            self.client = OpenAI(
                base_url=VLLM_URL_DICT[config.model_url].format(port=config.model_port),
                api_key=VLLM_API_KEY
            )
            self.model_path = model_name
            
    
    def generate(self, message):
        try:
            response = self.client.chat.completions.create(
                    model= self.model_path,
                    messages=message,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    seed=self.config.seed
                )
            output =  response.choices[0].message.content
        except Exception as e:
            # 处理其他可能的错误
            print(f"Error: {e}")
            output = ""
        # if 'usage' in generation_config:
        #     return output, usage
        # else:
        return output 