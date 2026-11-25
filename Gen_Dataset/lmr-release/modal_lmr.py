import modal
from pathlib import Path

ROOT = Path(__file__).resolve().parent

app = modal.App("lmr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgomp1")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "timm==0.6.13",
        "einops==0.8.0",
        "fvcore",
        "iopath",
        "yacs",
        "simplejson",
        "av",
        "pyyaml",
        "matplotlib",
        "termcolor",
        "tqdm",
        "psutil",
        "pandas",
        "scikit-learn",
        "tensorboard",
        "ffmpeg-python",
        "joblib",
    )
    .env({"PYTHONPATH": "/mnt"})
    .add_local_dir(ROOT / "slowfast", "/mnt/slowfast")
    .add_local_dir(ROOT / "configs", "/mnt/configs")
    .add_local_dir(ROOT / "tools", "/mnt/tools")
)


volume = modal.Volume.from_name("uit-adrone-lmr")

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=86400,
    cpu=4,
    memory=16384,
)
def runtime():
    pass
