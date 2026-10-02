# Use an official Python runtime as a parent image, optimized for 32-bit ARM (NanoPi/Raspberry Pi)
# For desktop testing (x86_64), Docker will automatically pull the corresponding x86_64 version.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install python packages
# Note: We use opencv-python-headless to avoid heavy GUI dependencies (X11) in the container
RUN pip install --no-cache-dir --upgrade pip && \
    grep -vE 'opencv-python' requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    pip install --no-cache-dir opencv-python-headless tflite-runtime --extra-index-url https://www.piwheels.org/simple

# Copy the rest of the application code
COPY . .

# Expose the web server port
EXPOSE 5000

# Run the native server when the container launches
CMD ["python", "native_server.py"]
