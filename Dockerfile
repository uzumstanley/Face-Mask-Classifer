# Use an official Python runtime as a parent image
FROM python:3.10.0

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Run streamlit when the container launches
CMD streamlit run FMaskCNN.py
