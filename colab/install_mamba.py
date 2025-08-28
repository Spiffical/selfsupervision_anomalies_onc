#!/usr/bin/env python3
import os
import sys
import subprocess

try:
    import torch
except Exception as e:
    print("[install_mamba] torch not importable:", e)
    sys.exit(1)

VERSION = os.environ.get("MAMBA_VER", "2.2.5")

if not torch.cuda.is_available():
    print("[install_mamba] Skipping: CUDA GPU not available.")
    print("  Tip: Runtime → Change runtime type → Hardware accelerator → GPU, then rerun.")
    sys.exit(0)

torch_mm = ".".join(torch.__version__.split(".")[:2])  # e.g., '2.4'
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
cu_major = "12" if (torch.version.cuda or "12").split(".")[0] == "12" else "11"

wheel = f"mamba_ssm-{VERSION}+cu{cu_major}torch{torch_mm}cxx11abi{abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
url = f"https://github.com/state-spaces/mamba/releases/download/v{VERSION}/{wheel}"

print("[install_mamba] Attempting prebuilt wheel:", url)
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", url])
    import mamba_ssm  # noqa: F401
    print("[install_mamba] ✅ Installed mamba-ssm:", mamba_ssm.__version__)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print("[install_mamba] No compatible prebuilt wheel found.")
    print("  torch:", torch.__version__, "cuda:", getattr(torch.version, 'cuda', None), "python:", sys.version)
    pass
except Exception as e:
    print("[install_mamba] Error during wheel install:", e)

if os.environ.get("ALLOW_BUILD", "0") == "1":
    print("[install_mamba] Building from source (ALLOW_BUILD=1)")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}{minor}"
        print("  TORCH_CUDA_ARCH_LIST=", os.environ["TORCH_CUDA_ARCH_LIST"])
    os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "2")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ninja"])  # speed up build
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"mamba-ssm=={VERSION}", "--no-build-isolation"])  # may compile
    import mamba_ssm  # noqa: F401
    print("[install_mamba] ✅ Installed mamba-ssm from source:", mamba_ssm.__version__)
else:
    print("[install_mamba] Skipping source build. Set ALLOW_BUILD=1 to allow compilation.")
    print("  Tip: Use Python 3.11 and Torch 2.4.1+cu121 for prebuilt wheels.")

