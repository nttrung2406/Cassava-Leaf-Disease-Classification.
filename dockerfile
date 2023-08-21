# Use an official PyTorch image as the base
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the code and data files into the container
COPY . /app
EXPOSE 8000
# Install required packages
RUN pip install pandas numpy scikit-learn opencv-python-headless matplotlib

# Run the Python script
CMD ["python", "classification.py"]
