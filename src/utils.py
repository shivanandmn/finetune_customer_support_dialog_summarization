from tqdm import tqdm


def trainable_parameter_count(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad():
            trainable_params += param.numel()
    return f"trainable params :{trainable_params}\t\ttotal_params :{all_params}"


