# https://hub.docker.com/_/python
FROM python:3.10-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY src/requirements.txt ./

# Set the module name
ENV MODULE auth

# Service must listen to $PORT environment variable.
ENV PORT 8080

#Upgrade pip
RUN pip install --upgrade pip

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

#Attempt to set up default en_core_web_sm. This is backup when our trained model is not working.
#Not sure if it is now working.
#RUN python3 -m venv .env
#RUN pip install -U pip setuptools wheel
#RUN python3 -m spacy download en_core_web_sm

# Install CORS so it can by pass the browser's CORS policy
RUN pip install -U Flask
RUN pip install -U flask-cors
# RUN pip install flask-stubs
# RUN pip install flask-cors-stubs
RUN pip install types-flask-cors

# Install Gunicorn
RUN pip install gunicorn

# Bundle app source
COPY __init__.py $APP_HOME/$MODULE/
COPY src $APP_HOME/$MODULE/src
#COPY tests $APP_HOME/$MODULE/tests

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 $MODULE.src.app:app