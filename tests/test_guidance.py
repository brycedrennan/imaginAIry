#
# import torch
#
# from imaginairy.utils import get_device
#
# torch.manual_seed(0)
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
#
# from tqdm.auto import tqdm, trange
# from torch import autocast
# import PIL.Image as PImage
# from PIL import Image
# import numpy
# from torchvision import transforms
# import torchvision.transforms.functional as f
# import random
# import requests
# from io import BytesIO
#
# # import clip
# import open_clip as clip
# from torch import nn
# import torch.nn.functional as F
# import io
#
# offload_device = "cpu"
# model_name = "CompVis/stable-diffusion-v1-4"
# attention_slicing = True #@param {"type":"boolean"}
# unet_path = False
#
# vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=True)
#
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# try:
#     text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", use_auth_token=True)
# except:
#     print("Text encoder could not be loaded from the repo specified for some reason, falling back to the vit-l repo")
#     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
#
# if unet_path!=None:
#     # unet = UNet2DConditionModel.from_pretrained(unet_path)
#     from huggingface_hub import hf_hub_download
#     model_name = hf_hub_download(repo_id=unet_path, filename="unet.pt")
#     unet = torch.jit.load(model_name)
# else:
#     unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=True)
#     if attention_slicing:
#         slice_size = unet.config.attention_head_dim // 2
#         unet.set_attention_slice(slice_size)
#
# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
#
# vae = vae.to(offload_device).half()
# text_encoder = text_encoder.to(offload_device).half()
# unet = unet.to(get_device()).half()
# class MakeCutouts(nn.Module):
#     def __init__(self, cut_size, cutn, cut_pow=1.):
#         super().__init__()
#         self.cut_size = cut_size
#         self.cutn = cutn
#         self.cut_pow = cut_pow
#
#     def forward(self, input):
#         sideY, sideX = input.shape[2:4]
#         max_size = min(sideX, sideY)
#         min_size = min(sideX, sideY, self.cut_size)
#         cutouts = []
#         for _ in range(self.cutn):
#             size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
#             offsetx = torch.randint(0, sideX - size + 1, ())
#             offsety = torch.randint(0, sideY - size + 1, ())
#             cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
#             cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
#         return torch.cat(cutouts)
#
#
# to_tensor_tfm = transforms.ToTensor()
#
#
# # mismatch of tons of image encoding / decoding / loading functions i can't be asked to clean up right now
#
# def pil_to_latent(input_im):
#     # Single image -> single latent in a batch (so size 1, 4, 64, 64)
#     with torch.no_grad():
#         with autocast("cuda"):
#             latent = vae.encode(to_tensor_tfm(input_im.convert("RGB")).unsqueeze(0).to(
#                 get_device()) * 2 - 1).latent_dist  # Note scaling
#     #   print(latent)
#     return 0.18215 * latent.mode()  # or .mean or .sample
#
#
# def latents_to_pil(latents):
#     # bath of latents -> list of images
#     latents = (1 / 0.18215) * latents
#     with torch.no_grad():
#         image = vae.decode(latents)
#     image = (image / 2 + 0.5).clamp(0, 1)
#     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#     images = (image * 255).round().astype("uint8")
#     pil_images = [Image.fromarray(image) for image in images]
#     return pil_images
#
#
# def get_latent_from_url(url, size=(512, 512)):
#     response = requests.get(url)
#     img = PImage.open(BytesIO(response.content))
#     img = img.resize(size).convert("RGB")
#     latent = pil_to_latent(img)
#     return latent
#
#
# def scale_and_decode(latents):
#     with autocast("cuda"):
#         # scale and decode the image latents with vae
#         latents = 1 / 0.18215 * latents
#         with torch.no_grad():
#             image = vae.decode(latents).sample.squeeze(0)
#         image = f.to_pil_image((image / 2 + 0.5).clamp(0, 1))
#         return image
#
#
# def fetch(url_or_path):
#     import io
#     if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
#         r = requests.get(url_or_path)
#         r.raise_for_status()
#         fd = io.BytesIO()
#         fd.write(r.content)
#         fd.seek(0)
#         return PImage.open(fd).convert('RGB')
#     return PImage.open(open(url_or_path, 'rb')).convert('RGB')
#
#
# """
# grabs all text up to the first occurrence of ':'
# uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
# """
#
#
# def split_weighted_subprompts(text, split=":"):
#     remaining = len(text)
#     prompts = []
#     weights = []
#     while remaining > 0:
#         if split in text:
#             idx = text.index(split)  # first occurrence from start
#             # grab up to index as sub-prompt
#             prompt = text[:idx]
#             remaining -= idx
#             # remove from main text
#             text = text[idx + 1:]
#             # find value for weight
#             if " " in text:
#                 idx = text.index(" ")  # first occurrence
#             else:  # no space, read to end
#                 idx = len(text)
#             if idx != 0:
#                 try:
#                     weight = float(text[:idx])
#                 except:  # couldn't treat as float
#                     print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
#                     weight = 1.0
#             else:  # no value found
#                 weight = 1.0
#             # remove from main text
#             remaining -= idx
#             text = text[idx + 1:]
#             # append the sub-prompt and its weight
#             prompts.append(prompt)
#             weights.append(weight)
#         else:  # no : found
#             if len(text) > 0:  # there is still text though
#                 # take remainder as weight 1
#                 prompts.append(text)
#                 weights.append(1.0)
#             remaining = 0
#     print(prompts, weights)
#     return prompts, weights
#
#
# # from some stackoverflow comment
# import numpy as np
#
#
# def lerp(a, b, x):
#     "linear interpolation"
#     return a + x * (b - a)
#
#
# def fade(t):
#     "6t^5 - 15t^4 + 10t^3"
#     return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
#
#
# def gradient(h, x, y):
#     "grad converts h to the right gradient vector and return the dot product with (x,y)"
#     vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
#     g = vectors[h % 4]
#     return g[:, :, 0] * x + g[:, :, 1] * y
#
#
# def perlin(x, y, seed=0):
#     # permutation table
#     np.random.seed(seed)
#     p = np.arange(256, dtype=int)
#     np.random.shuffle(p)
#     p = np.stack([p, p]).flatten()
#     # coordinates of the top-left
#     xi, yi = x.astype(int), y.astype(int)
#     # internal coordinates
#     xf, yf = x - xi, y - yi
#     # fade factors
#     u, v = fade(xf), fade(yf)
#     # noise components
#     n00 = gradient(p[p[xi] + yi], xf, yf)
#     n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
#     n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
#     n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
#     # combine noises
#     x1 = lerp(n00, n10, u)
#     x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
#     return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here
#
#
# def sample(args):
#     global in_channels
#     global text_encoder  # uugghhhghhghgh
#     global vae  # UUGHGHHGHGH
#     global unet  # .hggfkgjks;ldjf
#     # prompt = args.prompt
#     prompts, weights = split_weighted_subprompts(args.prompt)
#     h, w = args.size
#     steps = args.steps
#     scale = args.scale
#     classifier_guidance = args.classifier_guidance
#     use_init = len(args.init_img) > 1
#     if args.seed != -1:
#         seed = args.seed
#         generator = torch.manual_seed(seed)
#     else:
#         seed = random.randint(0, 10_000)
#         generator = torch.manual_seed(seed)
#     print(f"Generating with seed {seed}...")
#
#     # tokenize / encode text
#     tokens = [tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
#                         return_tensors="pt") for prompt in prompts]
#     with torch.no_grad():
#         # move CLIP to cuda
#         text_encoder = text_encoder.to(get_device())
#         text_embeddings = [text_encoder(tok.input_ids.to(get_device()))[0].unsqueeze(0) for tok in tokens]
#         text_embeddings = [text_embeddings[i] * weights[i] for i in range(len(text_embeddings))]
#         text_embeddings = torch.cat(text_embeddings, 0).sum(0)
#         max_length = 77
#         uncond_input = tokenizer(
#             [""], padding="max_length", max_length=max_length, return_tensors="pt"
#         )
#         uncond_embeddings = text_encoder(uncond_input.input_ids.to(get_device()))[0]
#         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#         # move it back to CPU so there's more vram for generating
#         text_encoder = text_encoder.to(offload_device)
#     images = []
#
#     if args.lpips_guidance:
#         import lpips
#         lpips_model = lpips.LPIPS(net='vgg').to(get_device())
#         init = to_tensor_tfm(fetch(args.init_img).resize(args.size)).to(get_device())
#
#     for batch_n in trange(args.batches):
#         with autocast("cuda"):
#             # unet = unet.to(get_device())
#             scheduler.set_timesteps(steps)
#             if not use_init or args.start_step == 0:
#                 latents = torch.randn(
#                     (1, in_channels, h // 8, w // 8),
#                     generator=generator
#                 )
#                 latents = latents.to(get_device())
#                 latents = latents * scheduler.sigmas[0]
#                 start_step = args.start_step
#             else:
#                 # Start step
#                 start_step = args.start_step - 1
#                 start_sigma = scheduler.sigmas[start_step]
#                 start_timestep = int(scheduler.timesteps[start_step])
#
#                 # Prep latents
#                 vae = vae.to(get_device())
#                 encoded = get_latent_from_url(args.init_img)
#                 if not classifier_guidance:
#                     vae = vae.to(offload_device)
#
#                 noise = torch.randn_like(encoded)
#                 sigmas = scheduler.match_shape(scheduler.sigmas[start_step], noise)
#                 noisy_samples = encoded + noise * sigmas
#
#                 latents = noisy_samples.to(get_device()).half()
#
#             if args.perlin_multi != 0:
#                 linx = np.linspace(0, 5, h // 8, endpoint=False)
#                 liny = np.linspace(0, 5, w // 8, endpoint=False)
#                 x, y = np.meshgrid(liny, linx)
#                 p = [np.expand_dims(perlin(x, y, seed=i), 0) for i in range(4)]  # reproducible seed
#                 p = np.concatenate(p, 0)
#                 p = torch.tensor(p).unsqueeze(0).cuda()
#                 latents = latents + (p * args.perlin_multi).to(get_device()).half()
#
#             for i, t in tqdm(enumerate(scheduler.timesteps), total=steps):
#                 if i > start_step:
#                     latent_model_input = torch.cat([latents] * 2)
#                     sigma = scheduler.sigmas[i]
#                     latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
#
#                     with torch.no_grad():
#                         # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
#                         # noise_pred = unet(latent_model_input, torch.tensor(t, dtype=torch.float32).cuda().half(), text_embeddings)#["sample"]
#                         noise_pred = unet(latent_model_input, torch.tensor(t, dtype=torch.float32).cuda(),
#                                           text_embeddings)  # ["sample"]
#
#                     # cfg
#                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                     noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
#
#                     # cg
#                     if classifier_guidance:
#                         # vae = vae.to(get_device())
#                         if vae.device != latents.device:
#                             vae = vae.to(latents.device)
#                         latents = latents.detach().requires_grad_()
#                         latents_x0 = latents - sigma * noise_pred
#                         denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
#                         if args.loss_scale != 0:
#                             loss = args.loss_fn(denoised_images) * args.loss_scale
#                         else:
#                             loss = 0
#                             init_losses = lpips_model(denoised_images, init)
#                             loss = loss + init_losses.sum() * args.lpips_scale
#
#                         cond_grad = -torch.autograd.grad(loss, latents)[0]
#                         latents = latents.detach() + cond_grad * sigma ** 2
#                         # vae = vae.to(offload_device)
#
#                     latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
#         vae = vae.to(get_device())
#         output_image = scale_and_decode(latents)
#         vae = vae.to(offload_device)
#         images.append(output_image)
#
#         import gc
#         gc.collect()
#         torch.cuda.empty_cache()
#
#         images[-1].save(f"output/{batch_n}.png")
#
#     return images
#
# def test_guided_image():
#     prompt = "tardigrade portrait [intricate] [artstation]"  # @param {"type":"string"}
#
#     # prompt = add_suffixes(prompt)
#
#     init_img = ""  # @param {"type":"string"}
#     size = [640, 640]  # @param
#     steps = 65  # @param
#     start_step = 0  # @param
#     perlin_multi = 0.4  # @param
#     scale = 7  # @param
#     seed = -1  # @param
#     batches = 4  # @param
#     # @markdown ---
#
#     # @markdown ### Classifier Guidance
#     # @markdown `classifier_guidance` is whether or not to use the loss function in the previous cell to guide the image (slows down image generation a lot) <br>
#     # @markdown it also is very hit-and-miss in terms of quality, but can be really really good, try setting batches high and then taking a nap <br>
#     # @markdown `lpips_guidance` is for if you're using an init_img, it'll let you start closer to the beginning while trying to keep the overall shapes similar
#     # @markdown `lpips_scale` is similar to `loss_scale` but it's how much to push the model to keep the shapes the same <br>
#     # @markdown `loss_scale` is how much to guide according to that loss function <br>
#     # @markdown `clip_text_prompt` is a prompt for CLIP to optimize towards, if using classifier guidance (supports weighting with `prompt:weight`) <br>
#     # @markdown `clip_image_prompt` is an image url for CLIP to optimize towards if using classifier guidance (supports weighting with `url|weight` because of colons coming up in urls) <br>
#     # @markdown for `clip_model_name` and `clip_model_pretrained` check out the openclip repository https://github.com/mlfoundations/open_clip <br>
#     # @markdown `cutn` is the amount of permutations of the image to show to clip (can help with stability) <br>
#     # @markdown `accumulate` is how many times to run the image through the clip model, can help if you can only fit low cutn on the machine <br>
#     # @markdown *you cannot use the textual inversion tokens with the clip text prompt*  <br>
#     # @markdown *also clip guidance sucks for most things except removing very small details that dont make sense*
#     classifier_guidance = True  # @param {"type":"boolean"}
#     lpips_guidance = False  # @param {"type":"boolean"}
#     lpips_scale = 0  # @param
#     loss_scale = 1.  # @param
#
#     class BlankClass():
#         def __init__(self):
#             bruh = 'BRUH'
#
#     args = BlankClass()
#     args.prompt = prompt
#     args.init_img = init_img
#     args.size = size
#     args.steps = steps
#     args.start_step = start_step
#     args.scale = scale
#     args.perlin_multi = perlin_multi
#     args.seed = seed
#     args.batches = batches
#     args.classifier_guidance = classifier_guidance
#     args.lpips_guidance = lpips_guidance
#     args.lpips_scale = lpips_scale
#     args.loss_scale = loss_scale
#
#     loss_scale = 1
#     # make_cutouts = MakeCutouts(224, 16)
#
#     clip_text_prompt = "tardigrade portrait [intricate] [artstation]"  # @param {"type":"string"}
#     # clip_text_prompt = add_suffixes(clip_text_prompt)
#     clip_image_prompt = ""  # @param {"type":"string"}
#
#     if loss_scale != 0:
#         # clip_model = clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(get_device())
#         clip_model_name = "ViT-B-32"  # @param {"type":"string"}
#         clip_model_pretrained = "laion2b_s34b_b79k"  # @param {"type":"string"}
#         clip_model, _, preprocess = clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_pretrained)
#         clip_model = clip_model.eval().requires_grad_(False).to(get_device())
#
#         cutn = 4  # @param
#         make_cutouts = MakeCutouts(clip_model.visual.image_size if type(clip_model.visual.image_size) != tuple else
#                                    clip_model.visual.image_size[0], cutn)
#
#     target = None
#     if len(clip_text_prompt) > 1:
#         clip_text_prompt, clip_text_weights = split_weighted_subprompts(clip_text_prompt)
#         target = clip_model.encode_text(clip.tokenize(clip_text_prompt).to(get_device())) * torch.tensor(
#             clip_text_weights).view(len(clip_text_prompt), 1).to(get_device())
#     if len(clip_image_prompt) > 1:
#         clip_image_prompt, clip_image_weights = split_weighted_subprompts(clip_image_prompt, split="|")
#         # pesky spaces
#         clip_image_prompt = [p.replace(" ", "") for p in clip_image_prompt]
#         images = [fetch(image) for image in clip_image_prompt]
#         images = [f.to_tensor(i).unsqueeze(0) for i in images]
#         images = [make_cutouts(i) for i in images]
#         encodings = [clip_model.encode_image(i.to(get_device())).mean(0) for i in images]
#
#         for i in range(len(encodings)):
#             encodings[i] = (encodings[i] * clip_image_weights[i]).unsqueeze(0)
#         # print(encodings.shape)
#         encodings = torch.cat(encodings, 0)
#         encoding = encodings.sum(0)
#
#         if target != None:
#             target = target + encoding
#         else:
#             target = encoding
#         target = target.half().to(get_device())
#
#     # free a little memory, we dont use the text encoder after this so just delete it
#     clip_model.transformer = None
#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()
#
#     def spherical_distance(x, y):
#         x = F.normalize(x, dim=-1)
#         y = F.normalize(y, dim=-1)
#         l = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
#         return l
#
#     def loss_fn(x):
#         with torch.autocast("cuda"):
#             cutouts = make_cutouts(x)
#             encoding = clip_model.encode_image(cutouts.float()).half()
#             loss = spherical_distance(encoding, target)
#             return loss.mean()
#
#     args.loss_fn = loss_fn
#
#
#     dtype = torch.float16
#     with torch.amp.autocast(device_type=get_device(), dtype=dtype):
#         output = sample(args)
#     print("Done!")
