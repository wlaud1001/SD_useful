"""
Reference
    - https://github.com/bloc97/CrossAttentionControl/blob/main/InverseCrossAttention_Release.ipynb
"""

import random
from difflib import SequenceMatcher

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm



@torch.no_grad()
def inverseSD(
    images,
    prompt,
    vae=None, unet=None, clip_tokenizer=None, clip_text_encoder=None,
    from_pretrained=None,
    guidance_scale=1.0,
    steps=50, 
    refine_iterations=0, 
    refine_strength=0.9, 
    refine_skip=0.7,
    width=512,
    height=512,
    generator=torch.cuda.manual_seed(798122),
    device='cuda'
):
    if from_pretrained is None:
        assert vae is not None, "vae is None"
        assert unet is not None, "unet is None"
        assert clip_tokenizer is not None, "clip_tokenizer is None"
        assert clip_text_encoder is not None, "clip_text_encoder is None"
        
        unet = unet.to(device)
        vae = vae.to(device)
        clip_text_encoder = clip_text_encoder.to(device)
        clip_tokenizer = clip_tokenizer.to(device)

    else:
        pipe = StableDiffusionPipeline.from_pretrained(from_pretrained, safety_checker=None, torch_dtype=torch.float16, use_auth_token=True).to(device)
        unet = pipe.unet
        vae = pipe.vae
        clip_text_encoder = pipe.text_encoder
        clip_tokenizer = pipe.tokenizer
    
    if isinstance(images, list):
        images = np.array([img.resize((width, height)) for img in images]).astype(np.float32) / 255.0 * 2.0 - 1.0
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        
        batch_size = len(images)
        uncond_prompt = [""] * batch_size  
        if not isinstance(prompt, list):
            prompt = [prompt] * batch_size
        else:
            assert len(prompt) == len(images), "prompt, images length"
            
    else:
        images = images.resize((width, height), resample=Image.Resampling.LANCZOS)
        images = np.array(images).astype(np.float32) / 255.0 * 2.0 - 1.0
        images = torch.from_numpy(images[np.newaxis, ...].transpose(0, 3, 1, 2))
        uncond_prompt = ""
        prompt = prompt


    #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if images.shape[1] > 3:
        images = images[:, :3] * images[:, 3:] + (1 - images[:, 3:])
    
    images = images.to(device)
    
    train_steps = 1000
    timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(device)
    
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
    alphas = torch.cumprod(1 - betas, dim=0)
    
    init_step = 0
    
    with autocast(device):
        init_latent = vae.encode(images).latent_dist.sample(generator=generator) * 0.18215
        tokens_unconditional = clip_tokenizer(uncond_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip_text_encoder(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip_text_encoder(tokens_conditional.input_ids.to(device)).last_hidden_state

        latent = init_latent

        for i in tqdm(range(steps), total=steps):
            t_index = i + init_step
            
            t = timesteps[t_index]
            t1 = timesteps[t_index + 1]
            #Magic number for tless taken from Narnia, used for backwards CFG correction
            tless = t - (t1 - t) * 0.25
            
            ap = alphas[t] ** 0.5
            bp = (1 - alphas[t]) ** 0.5
            ap1 = alphas[t1] ** 0.5
            bp1 = (1 - alphas[t1]) ** 0.5
            
            latent_model_input = latent
            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            #One reverse DDIM step
            px0 = (latent_model_input - bp * noise_pred) / ap
            latent = ap1 * px0 + bp1 * noise_pred
            
            #Initialize loop variables
            latent_refine = latent
            latent_orig = latent_model_input
            min_error = 1e10
            lr = refine_strength
            
            #Finite difference gradient descent method to correct for classifier free guidance, performs best when CFG is high
            #Very slow and unoptimized, might be able to use Newton's method or some other multidimensional root finding method
            if i > (steps * refine_skip):
                for k in range(refine_iterations):
                    #Compute reverse diffusion process to get better prediction for noise at t+1
                    #tless and t are used instead of the "numerically correct" t+1, produces way better results in practice, reason unknown...
                    noise_pred_uncond = unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
                    noise_pred_cond = unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    #One forward DDIM Step
                    px0 = (latent_refine - bp1 * noise_pred) / ap1
                    latent_refine_orig = ap * px0 + bp * noise_pred
                    
                    #Save latent if error is smaller
                    error = float((latent_orig - latent_refine_orig).abs_().sum())
                    if error < min_error:
                        latent = latent_refine
                        min_error = error

                    #print(k, error)
                    
                    #Break to avoid "overfitting", too low error does not produce good results in practice, why?
                    if min_error < 5:
                        break
                    
                    #"Learning rate" decay if error decrease is too small or negative (dampens oscillations)
                    if (min_error - error) < 1:
                        lr *= 0.9
                    
                    #Finite difference gradient descent
                    latent_refine = latent_refine + (latent_model_input - latent_refine_orig) * lr
    return latent
        