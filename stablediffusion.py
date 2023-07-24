import random
from difflib import SequenceMatcher

import numpy as np
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm


class StableDiffusion():
    def __init__(
        self,
        vae, unet, clip, clip_tokenizer, device='cuda'
    ):
        self.vae = vae
        self.unet = unet
        self.clip = clip
        self.clip_tokenizer = clip_tokenizer
        self.device = device

    @torch.no_grad()
    def sample(
        self, 
        prompt='',
        latent=None,
        prompt_edit=None, 
        prompt_edit_token_weights=[], 
        prompt_edit_tokens_range=(0, 1),
        prompt_edit_spatial_range=(0, 1),
        guidance_scale=7.5, steps=50, 
        scheduler=None,
        seed=None, 
        width=512, height=512, 
        init_image=None, init_image_strength=0.5, 
        vis_time_idx=25,
    ):
        width = width - width % 64
        height = height - height % 64

        if seed is None: seed = random.randrange(2**32 - 1)
        generator = torch.cuda.manual_seed(seed)

        if scheduler is None:
            scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        else:
            scheduler = scheduler
        scheduler.set_timesteps(steps)

        if init_image is not None:
            init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
            init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
            init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))
        
            #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])
                
            #Move image to GPU
            init_image = init_image.to(self.device)

            #Encode image
            with autocast(self.device):
                init_latent = self.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

            t_start = steps - int(steps * init_image_strength)
                
        else:
            init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
            t_start = 0
    
        #Generate random normal noise
        if latent is not None:
            noise = latent
        else:
            noise = torch.randn(init_latent.shape, generator=generator, device=self.device)
            
        latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=self.device)).to(self.device)
        
        #Process clip
        with autocast(self.device):
            tokens_unconditional = self.clip_tokenizer("", padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = self.clip(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            tokens_conditional = self.clip_tokenizer(prompt, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional = self.clip(tokens_conditional.input_ids.to(self.device)).last_hidden_state
            
            self.init_attention_func()
            self.init_attention_weights(prompt_edit_token_weights)

            #Process prompt editing
            if prompt_edit is not None:
                tokens_conditional_edit = self.clip_tokenizer(prompt_edit, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional_edit = self.clip(tokens_conditional_edit.input_ids.to(self.device)).last_hidden_state
                
                self.init_attention_edit(tokens_conditional, tokens_conditional_edit)
                
            timesteps = scheduler.timesteps[t_start:]
            
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                #sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                #Predict the unconditional noise residual
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
                #Prepare the Cross-Attention layers
                if i == vis_time_idx:
                    self.save_last_tokens_attention()
                    
                if prompt_edit is not None:
                    self.save_last_tokens_attention()
                    self.save_last_self_attention()
                else:
                    #Use weights on non-edited prompt when edit is None
                    self.use_last_tokens_attention_weights()
                    
                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                
                #Edit the Cross-Attention layer activations
                if prompt_edit is not None:
                    t_scale = 1 - (t / scheduler.num_train_timesteps)
                    if t_scale >= prompt_edit_tokens_range[0] and t_scale <= prompt_edit_tokens_range[1]:
                        self.use_last_tokens_attention()

                    if t_scale >= prompt_edit_spatial_range[0] and t_scale <= prompt_edit_spatial_range[1]:
                        self.use_last_self_attention()
                        
                    #Use weights on edited prompt
                    self.use_last_tokens_attention_weights()

                    #Predict the edited conditional noise residual using the cross-attention masks
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample
                    
                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t, latent).prev_sample

            #scale and decode the image latents with vae
            latent = latent / 0.18215
            image = self.vae.decode(latent.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)

    @torch.no_grad()
    def inverse(
        self,
        init_image, 
        prompt="", 
        guidance_scale=3.0, 
        steps=50, 
        refine_iterations=3, 
        refine_strength=0.9, 
        refine_skip=0.7,
        width=512,
        height=512,
    ):
        #Change size to multiple of 64 to prevent size mismatches inside model
        
        if isinstance(init_image, list):
            init_image = np.array([img.resize((width, height)) for img in init_image]).astype(np.float32) / 255.0 * 2.0 - 1.0
            init_image = torch.from_numpy(init_image.transpose(0, 3, 1, 2))
            batch_size = len(init_image)
            uncond_prompt = [""] * batch_size  
            if not isinstance(prompt, list):
                prompt = [prompt] * batch_size
                
        else:
            init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
            init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
            init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))


            
        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        #Move image to GPU
        init_image = init_image.to(self.device)

        train_steps = 1000
        step_ratio = train_steps // steps
        timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(self.device)
        
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
        alphas = torch.cumprod(1 - betas, dim=0)
        
        init_step = 0

        #Fixed seed such that the vae sampling is deterministic, shouldn't need to be changed by the user...
        generator = torch.cuda.manual_seed(798122)

        #Process clip
        with autocast(self.device):
            init_latent = self.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
            tokens_unconditional = self.clip_tokenizer(uncond_prompt, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = self.clip(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            tokens_conditional = self.clip_tokenizer(prompt, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional = self.clip(tokens_conditional.input_ids.to(self.device)).last_hidden_state
            
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
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                
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
                        noise_pred_uncond = self.unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
                        noise_pred_cond = self.unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
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

    def init_attention_func(self):
        #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276

        class NewAttnProcessor:
            r"""
            Default processor for performing attention-related computations.
            """
            def __call__(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states
        
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
        
                input_ndim = hidden_states.ndim
        
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
                query = attn.to_q(hidden_states)
        
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
        
                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
        
                attn_slice = attn.get_attention_scores(query, key, attention_mask)

                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                        
                        attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        attn_slice = self.last_attn_slice
        
                    self.use_last_attn_slice = False
        
                if self.save_last_attn_slice:
                    self.last_attn_slice = attn_slice
                    self.save_last_attn_slice = False
        
                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    attn_slice = attn_slice * self.last_attn_slice_weights
                    self.use_last_attn_weights = False

                hidden_states = torch.bmm(attn_slice, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
        
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
        
                hidden_states = hidden_states / attn.rescale_output_factor
        
                return hidden_states

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention":
                module.processor = NewAttnProcessor()
                module.processor.last_attn_slice = None
                module.processor.use_last_attn_slice = False
                module.processor.use_last_attn_weights = False
                module.processor.save_last_attn_slice = False

    def init_attention_weights(self, weight_tuples):
        tokens_length = self.clip_tokenizer.model_max_length
        weights = torch.ones(tokens_length)
        
        for i, w in weight_tuples:
            if i < tokens_length and i >= 0:
                weights[i] = w
        
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn2" in name:
                module.processor.last_attn_slice_weights = weights.to(self.device)
            if module_name == "Attention" and "attn1" in name:
                module.processor.last_attn_slice_weights = None

    def init_attention_edit(self, tokens, tokens_edit):
        tokens_length = self.clip_tokenizer.model_max_length
        mask = torch.zeros(tokens_length)
        indices_target = torch.arange(tokens_length, dtype=torch.long)
        indices = torch.zeros(tokens_length, dtype=torch.long)

        tokens = tokens.input_ids.numpy()[0]
        tokens_edit = tokens_edit.input_ids.numpy()[0]

        for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
            if b0 < tokens_length:
                if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                    mask[b0:b1] = 1
                    indices[b0:b1] = indices_target[a0:a1]

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn2" in name:
                module.processor.last_attn_slice_mask = mask.to(self.device)
                module.processor.last_attn_slice_indices = indices.to(self.device)
            if module_name == "Attention" and "attn1" in name:
                module.processor.last_attn_slice_mask = None
                module.processor.last_attn_slice_indices = None

    def use_last_tokens_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn2" in name:
                module.processor.use_last_attn_slice = use
                
    def use_last_tokens_attention_weights(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn2" in name:
                module.processor.use_last_attn_weights = use
                
    def use_last_self_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn1" in name:
                module.processor.use_last_attn_slice = use
                
    def save_last_tokens_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn2" in name:
                module.processor.save_last_attn_slice = save
                
    def save_last_self_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and "attn1" in name:
                module.processor.save_last_attn_slice = save
