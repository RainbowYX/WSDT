from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
from register_attention import *
import os
from diffusers.utils import load_image
import numpy as np
import argparse

# load model
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
    scheduler=scheduler
).to("cuda")
print("pipeline loaded")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_image",type=str,default="examples/van gogh painting house.jpg",help="path to the input image")
    parser.add_argument("--cont_prompt",type=str,default="",help="A bear in van gogh painting style") 
    parser.add_argument("--sty_prompt",type=str,default="",help="van gogh painting house") 
    parser.add_argument("--steps",type=int,default=50,help="inference_steps")
    parser.add_argument("--cfs",type=float,default=7.5,help="guidance_scale")
    parser.add_argument("--thres",type=float,default=0.2,help="threshold")
    parser.add_argument('--output_path', type=str, default="./output" , help='path to the output')
    args=parser.parse_args()
    os.makedirs(args.output_path,exist_ok=True)
    x0 = np.array(load_image(args.input_image).resize((1024, 1024)))
    zts = forward_noisy(pipeline, x0, args.steps)
    inversion_callback = make_inversion_callback(zts)

    prompts = [args.sty_prompt,args.cont_prompt]
    latents = torch.randn((len(prompts), 4, 128, 128), device='cuda', generator=torch.Generator(device="cuda").   manual_seed(0),dtype=pipeline.unet.dtype)
    latents[0] = zts[0]
    latents=adain(latents,2)
    register_attention_processors(pipeline, args.thres)
    images = pipeline(prompts, latents=latents,
                      callback_on_step_end=inversion_callback,
                      num_inference_steps=args.steps, guidance_scale=args.cfs).images
    images[0].save(os.path.join(args.output_path,"style.png"))
    images[1].save(os.path.join(args.output_path,"generated_image.png"))