# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirenments.txt .

# Install the dependencies
RUN pip install -r requirenments.txt

# Copy the entire project into the container
COPY . .

# Define the default command to run your Python script
CMD ["bash", "-c", "python src/data/load.py && python src/data/preprocesing.py && python src/data/eda.py && python src/data/feature.py && python src/model/train_evaluation.py && python src/model/predict.py"]
