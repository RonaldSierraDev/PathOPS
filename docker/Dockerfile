# --- Stage 1: build the operations console ---
# Node exists only in this stage; the runtime image below gets the built
# assets without npm or node_modules coming along with them.
FROM node:20-slim AS console

WORKDIR /console

# Copy the manifests alone first so `npm ci` is cached against dependency
# changes rather than re-running on every source edit.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Empty base URL: the console is served by the same FastAPI process it calls,
# so requests resolve to relative paths on the same origin. Leaving this unset
# would bake in the localhost:8000 dev default and break the deployed console.
ENV VITE_API_URL=""
RUN npm run build

# --- Stage 2: the API, which also serves those assets ---
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

# Matches FRONTEND_DIST's default in pathml.api.main, which is relative to
# this WORKDIR.
COPY --from=console /console/dist frontend/dist

EXPOSE 8000

CMD ["uvicorn", "pathml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
