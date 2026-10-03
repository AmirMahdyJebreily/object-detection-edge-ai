# Use Ubuntu 16.04 which has glibc 2.23 and Python 3.5
FROM ubuntu:16.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV 2.4.9 and Python 3.5
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install python packages (filter out opencv-python as we get it from apt)
RUN pip3 install --no-cache-dir --upgrade "pip<21" setuptools && \
    grep -vE 'opencv-python' requirements.txt > requirements_filtered.txt && \
    pip3 install --no-cache-dir -r requirements_filtered.txt

# Copy the rest of the application code
COPY . .

# Expose the web server port
EXPOSE 5000

# Run the native server when the container launches
CMD ["python3", "native_server.py"]
