# Use a standard RunPod image with PyTorch and CUDA pre-installed
FROM runpod/pytorch:2.1.0-cuda11.8.0-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install the Python packages listed in your requirements file
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your project files (like runpod.py)
COPY . .

# Command that RunPod will use to start your worker
CMD ["python", "-u", "runpod.py"]