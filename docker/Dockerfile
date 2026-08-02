FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

# Install the CPU-only torch/torchvision build first, from PyTorch's own CPU
# wheel index -- otherwise `pip install .` below pulls the default PyPI
# wheels, which bundle CUDA/cuDNN/NCCL and inflate the image by ~6GB for a
# Fargate task that has no GPU and never will.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "pathml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
