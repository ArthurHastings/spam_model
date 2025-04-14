FROM python:3.12.9

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Upgrade pip
RUN pip install --upgrade pip

# Install MLflow and Cloudpickle
RUN pip install mlflow==2.21.2 cloudpickle==3.0.0  # Adjust cloudpickle if needed

# Install missing dependencies
# RUN pip install --no-cache-dir attrs==25.1.0 defusedxml==0.7.1 dill==0.3.8 google-cloud-storage==3.1.0 \
#     huggingface-hub==0.29.3 ipython==8.32.0 matplotlib==3.10.0 numpy==2.0.2 \
#     opencv-python==4.11.0.86 pandas==2.2.3 psutil==7.0.0 pydot==3.0.4 \
#     scikit-learn==1.6.1 scipy==1.15.2 tensorflow  # Use generic tensorflow

pip install --no-cache-dir attrs==25.1.0
pip install --no-cache-dir cloudpickle==3.1.1
pip install --no-cache-dir defusedxml==0.7.1
pip install --no-cache-dir dill==0.3.8
pip install --no-cache-dir huggingface
pip install --no-cache-dir ipython==8.32.0
pip install --no-cache-dir matplotlib==3.10.0
pip install --no-cache-dir numpy==1.26.4
pip install --no-cache-dir pandas==2.2.3
pip install --no-cache-dir psutil==7.0.0
pip install --no-cache-dir scikit-learn==1.6.1
pip install --no-cache-dir scipy==1.13.1
pip install --no-cache-dir tensorflow==2.16.1

# Set working directory
WORKDIR /app

# Copy models
COPY ./models /app/models

# Expose MLflow port
EXPOSE 5001

# Start MLflow model serving
CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5001", "--model-uri", "file:///app/models/spam_detector_raw", "--no-conda"]