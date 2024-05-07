from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image

import torch
from enum import Enum

class Pipeline(Enum):
    SDXL_PIPELINE = 1
    SD_PIPELINE = 2
    KANDINSKY = 2

def get_pipeline(pipeline_type: Pipeline, seed: int):
    pipeline = None
    generator = torch.Generator("cuda").manual_seed(seed)

    if pipeline_type == Pipeline.SD_PIPELINE:
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

    elif pipeline_type == Pipeline.SDXL_PIPELINE:
        pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
    
    elif pipeline_type == Pipeline.KANDINSKY:
        pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
        
    else:
        raise Exception("Pipeline not supported")

    return pipeline, generator

