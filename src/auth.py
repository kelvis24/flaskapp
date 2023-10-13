# Imports
from flask import Flask, request

# Create the Flask app
app = Flask(__name__)

# Define the root endpoints


@app.route('/')
def index():
    return '<h1> VERIFY API </h1>'

# Define the /verify endpoint


@app.route('/verify/<code>')
def verify(code):
    return code  # This returns the code from the URL

# Define a ping endpoint


@app.route("/ping")
def ping():
    return "pong"


# Run the app if this script is the main entry point
if __name__ == '__main__':
    app.run()
