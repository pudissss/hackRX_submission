# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application's code into the container at /code
COPY . /code/app

# Tell Docker that the container listens on port 7860
EXPOSE 7860

# Command to run the application using Gunicorn (a production-ready server)
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:7860", "app.app:app"]
