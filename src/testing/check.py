import torch

def setup_torch(verbose: bool = True):
    if torch.cuda.is_available():
        try:
            torch.set_default_device("cuda")
            
            if verbose:
                print("✅ Using CUDA globally")
                print("GPU:", torch.cuda.get_device_name(0))
            
            return torch.device("cuda")
        except:
            pass

    torch.set_default_device("cpu")
    
    if verbose:
        print("❌ Using CPU globally")
    
    return torch.device("cpu")
