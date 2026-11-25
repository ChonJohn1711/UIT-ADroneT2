
import modal

app = modal.App("lmr-gpu-test")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1")
)

@app.function(
    image=image,
    gpu="A10",
    timeout=300,
)
def test_gpu():
    import torch
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    assert torch.cuda.is_available()

    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("[PASS] Modal GPU computation")

@app.local_entrypoint()
def main():
    test_gpu.remote()
