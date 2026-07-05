import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_lora_to_linear(model, target_layer_name, r=2, a=4): 
    layer_found = False

    for name, module in model.named_modules():
        if name == target_layer_name:
            layer_found = True
            
            input_size = module.in_features
            output_size = module.out_features
            
            weight_device = module.weight.device
            weight_dtype = module.weight.dtype

            for param in module.parameters():
                param.requires_grad_(False)

            A = nn.Parameter(torch.zeros(output_size, r, device=weight_device, dtype=weight_dtype, requires_grad=True))
            B = nn.Parameter(torch.randn(r, input_size, device=weight_device, dtype=weight_dtype, requires_grad=True))
            
            module.register_parameter('A', A)
            module.register_parameter('B', B)
            
            module.lora_a = a
            module.lora_r = r

            def make_lora(mod, inputs):
                x = inputs[0] 
                delta_W = torch.matmul(mod.A, mod.B)
                delta_W = delta_W * (mod.lora_a / mod.lora_r)
                mod.lora_out = F.linear(x, delta_W)
                return inputs

            def lora_forward(mod, input, output):
                return output + mod.lora_out

            module.register_forward_pre_hook(make_lora)
            module.register_forward_hook(lora_forward)
            
            break 
            
    if not layer_found:
        print(f"Warning: Layer '{target_layer_name}' was not found in the model!")
            
    return model