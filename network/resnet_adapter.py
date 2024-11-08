import torch.nn as nn
from collections import OrderedDict


class AdapterWrapperResNet(nn.Module):
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
        target_conv = self.resnet.conv1
        # adapter = adapter_class(
        #     r=gamma,
        #     lora_alpha=lora_alpha,
        #     conv_layer=target_conv
        # )
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
        
        setattr(self.resnet, "conv1", adapter)

        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2"]:
                    target_conv = getattr(bottleneck_layer, cv)
                    # adapter = adapter_class(
                    #     r=gamma,
                    #     lora_alpha=lora_alpha,
                    #     conv_layer=target_conv
                    # )
                    adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
                    setattr(bottleneck_layer, cv, adapter)


    def add_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        target_conv = self.resnet.conv1
        adapter = adapter_class(
            r=gamma,
            lora_alpha=lora_alpha,
            conv_layer=target_conv
        )
        setattr(self.resnet, "conv1", adapter)

        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2"]:
                    target_conv = getattr(bottleneck_layer, cv)
                    adapter = adapter_class(
                        r=gamma,
                        lora_alpha=lora_alpha,
                        conv_layer=target_conv
                    )
                    setattr(bottleneck_layer, cv, adapter)


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
