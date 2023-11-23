# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY src/requirements.txt ./
COPY src/ppe.pt ./

# Set the module name
ENV MODULE app
ENV QT_X11_NO_MITSHM=1

# Service must listen to $PORT environment variable.
ENV PORT 8080

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies required for psutil and OpenCV
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Install CORS so it can bypass the browser's CORS policy
RUN pip install -U Flask flask-cors types-flask-cors

# Bundle app source
COPY __init__.py $APP_HOME/$MODULE/
COPY src $APP_HOME/$MODULE/src
#COPY tests $APP_HOME/$MODULE/tests

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 $MODULE.src.app:app
