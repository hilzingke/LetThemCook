from flask import Flask
import os
import argparse

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is a simple Flask app with a custom port!"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get("FLASK_PORT", 5000)),
                        help='Port number to run the Flask app on')
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)