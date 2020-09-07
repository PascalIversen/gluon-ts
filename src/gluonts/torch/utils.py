import torch


def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
