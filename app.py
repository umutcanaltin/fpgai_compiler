import argparse
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return 1

@app.route('/open_model', methods=['POST'])
def open_model():
    onnx_file = request.files['file']
    return 1

@app.route('/generate_hls', methods=['POST'])
def generate_hls_files():
    return 1

@app.route('/synt_hls', methods=['POST'])
def synt_hls_files():
    return 1

@app.route('/hls_export_rtl', methods=['POST'])
def export_rtl_from_hls():
    return 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='the hostname to listen on. \
                              Set this to "0.0.0.0" to have the server available externally as well')
    parser.add_argument('--port', type=int, default=5000,
                        help='the port of the webserver. Defaults to 5000.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()