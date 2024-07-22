import torch.nn as nn
from collections import OrderedDict
import torch
from mmcv.cnn import ConvModule
from peft.lora_fast_shufflenet import LoraConv2d, MultiLoraConv2d


def mmcv_conv_forward(self,
            input_x,
            activate: bool = True,
            norm: bool = True) -> torch.Tensor:
    x, alphas = input_x[0], input_x[1]
    layer_index = 0
    while layer_index < len(self.order):
        layer = self.order[layer_index]
        if layer == 'conv':
            if self.with_explicit_padding:
                x = self.padding_layer(x)
            # if the next operation is norm and we have a norm layer in
            # eval mode and we have enabled `efficient_conv_bn_eval` for
            # the conv operator, then activate the optimized forward and
            # skip the next norm operator since it has been fused
            if layer_index + 1 < len(self.order) and \
                    self.order[layer_index + 1] == 'norm' and norm and \
                    self.with_norm and not self.norm.training and \
                    self.efficient_conv_bn_eval_forward is not None:
                self.conv.forward = partial(
                    self.efficient_conv_bn_eval_forward, self.norm,
                    self.conv)
                layer_index += 1
                x = self.conv(x, alphas)
                del self.conv.forward
            else:
                if self.conv.__class__ == MultiLoraConv2d:
                    # print(self.conv.__class__, x.device, self.conv.conv.weight.device)
                    x = self.conv({0:x.to(self.conv.conv.weight.device), 1:alphas.to(self.conv.conv.weight.device)})
                else:
                    # print(self.conv.__class__, x.device, self.conv.weight.device)
                    x = self.conv(x.to(self.conv.weight.device))
        elif layer == 'norm' and norm and self.with_norm:
            x = self.norm(x)
        elif layer == 'act' and activate and self.with_activation:
            x = self.activate(x)
        layer_index += 1
    return {0:x, 1:alphas}

class AdapterWrapperShuffleNet(nn.Module):
    def __init__(self, resnet_model, adapter_class, num_task, gamma, lora_alpha):
        super().__init__()
        self.resnet = resnet_model
        self.add_multi_adapter(adapter_class, num_task, gamma, lora_alpha)
        self.model_frozen = False
        # self.freeze_model(True)


    def add_multi_adapter(self, adapter_class, num_task, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        target_conv = self.resnet.conv1.conv
        # bound_method = mmcv_conv_forward.__get__(self.resnet.conv1, self.resnet.conv1.__class__)
        # assert self.resnet.conv1.__class__ == ConvModule
        # setattr(self.resnet.conv1, 'forward', bound_method)

        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
        
        if target_conv.groups == 1:
            # setattr(target_conv, "conv1", adapter)
            setattr(self.resnet.conv1, "conv", adapter)
        
        # Add adapter for resnet blocks
        target_layers = self.resnet.layers

        for th, layer in enumerate(target_layers):
            # print('layer:', layer)
            if th == len(target_layers)-1:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=layer.conv, num_task=num_task)
                if layer.conv.groups == 1:
                    setattr(layer, 'conv', adapter)
                break
            for bottleneck_layer in layer:
                # print('bottleneck_layer:', bottleneck_layer)
                if hasattr(bottleneck_layer, 'branch1'):
                    for each_branch in [bottleneck_layer.branch1, bottleneck_layer.branch2]:
                        # print('each_branch:', each_branch)
                        for each_conv in each_branch:
                            # print('each_conv:', each_conv)
                            # bound_method = mmcv_conv_forward.__get__(each_conv, each_conv.__class__)
                            # assert each_conv.__class__ == ConvModule
                            # setattr(each_conv, 'forward', bound_method)

                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, num_task=num_task)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)
                else:
                    for each_branch in [bottleneck_layer.branch2]:
                        # print('each_branch:', each_branch)
                        for each_conv in each_branch:
                            # print('each_conv:', each_conv)
                            # bound_method = mmcv_conv_forward.__get__(each_conv, each_conv.__class__)
                            # assert each_conv.__class__ == ConvModule
                            # setattr(each_conv, 'forward', bound_method)

                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, num_task=num_task)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)

        # raise ValueError()

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.resnet, True)
        other_param_num = count_parameters(self.resnet, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = (ratio / (1 - ratio))
        print("Ratio:", final_ratio)

        return final_ratio


    def forward(self, x, alphas=None):
        return self.resnet(x, alphas=alphas)


    def freeze_model(self, freeze=True): # 
        """Freezes all weights of the model."""
        if freeze: # 只更新lora, 非fc中的bias, 以及bn
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze


    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()

        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bn" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                if "fc" not in name:
                    adapter_dict[name] = param
        return adapter_dict


def create_pretrained_classifier(classifier_name):
    """
    create pretrained classifier from classifier_name and load publicly available model weights.

    :param classifier_name: a string classifier name. (ResNet50, ResNet18, ResNet152, DEIT supported)
    :return: a nn.Module class classifier.
    """
    if classifier_name == "ResNet50":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        classifier = resnet50(weights=weights)
    elif classifier_name ==" ResNet18":
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        classifier = resnet18(weights=weights)
    else:
        raise ValueError(f"classifier_name is not supported for : {classifier_name}")

    return classifier


if __name__ == "__main__":
    from peft.lora import LoraConv2d
    from guided_diffusion.script_util import create_pretrained_classifier
    import torch
    adapter_class = LoraConv2d
    resnet = create_pretrained_classifier(classifier_name="ResNet18")
    adapter_resnet = AdapterWrapperResNet(resnet, adapter_class, gamma=8, lora_alpha=8)

    adapter_resnet.freeze_model(True)
    adapter_resnet.calculate_training_parameter_ratio()

    state_dict = adapter_resnet.adapter_state_dict()
    torch.save(state_dict, "/tmp/ss.ckpt")
    adapter_resnet.load_state_dict(torch.load("/tmp/ss.ckpt"), strict=False)
