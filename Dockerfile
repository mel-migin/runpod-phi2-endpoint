# Use a more fundamental, stable base image from RunPod
FROM runpod/base:0.4.0-cuda11.8.0

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install PyTorch, then install the rest of the requirements
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Copy the rest of your project files (like runpod.py)
COPY . .

# Command that RunPod will use to start your worker
CMD ["python", "-u", "runpod.py"]