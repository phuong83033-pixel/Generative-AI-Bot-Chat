import sys
import subprocess
from pathlib import Path

def print_result(name: str, passed: bool, details: str = ""):
    status = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    detail_str = f" - {details}" if details else ""
    print(f"{status} {name}{detail_str}")

def check_python_version():
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 10
    print_result("Python Version >= 3.10", passed, f"{version.major}.{version.minor}.{version.micro}")

def check_cuda_pytorch():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            details = f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {device_name} ({vram:.1f} GB VRAM)"
            print_result("CUDA & PyTorch", True, details)
        else:
            print_result("CUDA & PyTorch", False, f"PyTorch {torch.__version__} installed, but CUDA not available.")
    except ImportError:
        print_result("CUDA & PyTorch", False, "PyTorch not installed.")

def check_sounddevice():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        
        in_name = devices[default_in]['name'] if default_in is not None and default_in >= 0 else "None"
        out_name = devices[default_out]['name'] if default_out is not None and default_out >= 0 else "None"
        
        print_result("sounddevice (Audio)", True, f"In: {in_name}, Out: {out_name}")
    except ImportError:
        print_result("sounddevice (Audio)", False, "sounddevice not installed.")
    except Exception as e:
        print_result("sounddevice (Audio)", False, f"Error querying devices: {e}")

def check_packages():
    packages = ["faster_whisper", "llama_cpp", "edge_tts"]
    for pkg in packages:
        try:
            __import__(pkg)
            print_result(f"Package '{pkg}'", True)
        except ImportError:
            print_result(f"Package '{pkg}'", False, "Not installed.")

def check_configs():
    paths = ["data/logs", "models"]
    for p in paths:
        path = Path(p)
        if path.exists():
            print_result(f"Directory '{p}'", True)
        else:
            print_result(f"Directory '{p}'", False, "Does not exist (will be created if needed).")

def main():
    print("Running Environment Checks...\n")
    check_python_version()
    check_cuda_pytorch()
    check_sounddevice()
    check_packages()
    check_configs()
    print("\nChecks complete.")

if __name__ == "__main__":
    main()
