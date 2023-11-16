import torch
from transformers import CLIPTextModelWithProjection

from imaginairy.model_manager import get_diffusion_model
from imaginairy.utils import get_device
from imaginairy.weight_management import utils


def trace_execution_order(module, args, func_name=None):
    """
    Trace the execution order of a torch module and store full hierarchical state_dict paths.
    :param module: The module to trace.
    :param args: The arguments to pass to the module.
    :return: A list of full hierarchical state_dict paths in the order they were used.
    """
    execution_order = []

    hooks = []

    def add_hooks(module, prefix=""):
        for name, submodule in module.named_children():
            # Construct the hierarchical name
            module_full_name = f"{prefix}.{name}" if prefix else name

            def log(mod, inp, out, module_full_name=module_full_name):
                hook(mod, module_full_name)

            hooks.append(submodule.register_forward_hook(log))

            # Recursively add hooks to all child modules
            add_hooks(submodule, module_full_name)

    def hook(module, module_full_name):
        # Retrieve state_dict and iterate over its items to get full paths
        for name, param in module.named_parameters(recurse=False):
            full_path = f"{module_full_name}.{name}"
            execution_order.append(full_path)
        for name, buffer in module.named_buffers(recurse=False):
            print(name)
            full_path = f"{module_full_name}.{name}"
            execution_order.append(full_path)

    # Initialize hooks
    add_hooks(module)

    # Execute the module
    with torch.no_grad():
        if func_name is not None:
            getattr(module, func_name)(*args)
        else:
            module(*args)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return execution_order


def trace_compvis_execution_order(device=None):
    model = get_diffusion_model()._mmmw_load_model()

    # vae
    image_size = 256
    img_in = torch.randn(1, 3, image_size, image_size).to(get_device())
    vae_execution_order = trace_execution_order(
        model.first_stage_model, (img_in,), func_name="encode_all_at_once"
    )

    latent_in = torch.randn(1, 4, 32, 32).to(get_device())
    vae_execution_order.extend(
        trace_execution_order(model.first_stage_model, (latent_in,), func_name="decode")
    )
    # text encoder model
    text = "hello"
    text_execution_order = trace_execution_order(model.cond_stage_model, (text,))

    # unet
    latent_in = torch.randn(1, 4, 32, 32).to(get_device())
    text_embedding = [torch.randn(1, 77, 768).to(get_device())]
    timestep = torch.tensor(data=[0]).to(get_device())
    unet_execution_order = trace_execution_order(
        model.model, (latent_in, timestep, text_embedding, text_embedding)
    )

    return vae_execution_order, text_execution_order, unet_execution_order


def trace_sd15_diffusers_execution_order(device=None):
    from diffusers import AutoencoderKL, UNet2DConditionModel

    if device is None:
        device = get_device()

    # vae
    image_size = 256
    img_in = torch.randn(1, 3, image_size, image_size).to(device)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device)
    vae_execution_order = trace_execution_order(vae, (img_in,))

    # text encoder model

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
    ).to(device)
    tokens = torch.Tensor(
        [
            [
                49406,
                3306,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
            ]
        ]
    )
    tokens = tokens.to(device).to(torch.int64)

    text_execution_order = trace_execution_order(text_encoder, (tokens,))

    # unet
    latent_in = torch.randn(1, 4, 32, 32).to(device)
    text_embedding = torch.randn(1, 77, 768).to(device)
    timestep = torch.tensor(data=[0]).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device)
    unet_execution_order = trace_execution_order(
        unet, (latent_in, timestep, text_embedding)
    )

    return vae_execution_order, text_execution_order, unet_execution_order


def calc_and_save_compvis_traces():
    model_name = "stable-diffusion-1-5"
    format_name = "compvis"

    (
        vae_execution_order,
        text_execution_order,
        unet_execution_order,
    ) = trace_compvis_execution_order()

    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="vae",
        execution_order=vae_execution_order,
    )
    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="text",
        execution_order=text_execution_order,
    )
    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="unet",
        execution_order=unet_execution_order,
    )


def calc_and_save_sd15_diffusers_traces():
    model_name = "stable-diffusion-1-5"
    format_name = "diffusers"

    (
        vae_execution_order,
        text_execution_order,
        unet_execution_order,
    ) = trace_sd15_diffusers_execution_order()

    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="vae",
        execution_order=vae_execution_order,
    )
    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="text",
        execution_order=text_execution_order,
    )
    process_execution_order(
        model_name=model_name,
        format_name=format_name,
        component_name="unet",
        execution_order=unet_execution_order,
    )


def process_execution_order(model_name, component_name, format_name, execution_order):
    prefixes = utils.prefixes_only(execution_order)
    utils.save_model_info(
        model_name,
        component_name,
        format_name,
        "prefix-execution-order",
        prefixes,
    )


if __name__ == "__main__":
    calc_and_save_sd15_diffusers_traces()
    calc_and_save_compvis_traces()
