import os
import socket
import subprocess
import sys
import time
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(SCRIPT_DIR, 'fake_news_detection', 'app.py')


def find_free_port(start=8501, end=8700):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError('No available port found between 8501 and 8700')

if __name__ == '__main__':
    print('Starting Streamlit app...')
    port = find_free_port()
    try:
        process = subprocess.Popen([
            sys.executable,
            '-m',
            'streamlit',
            'run',
            APP_PATH,
            '--server.port',
            str(port),
            '--server.headless',
            'false',
        ], cwd=SCRIPT_DIR)
        url = f'http://localhost:{port}'
        time.sleep(5)
        webbrowser.open(url)
        print(f'The app should open in your default browser at {url}.')
        process.wait()
    except KeyboardInterrupt:
        print('\nStreamlit process interrupted by user.')
    finally:
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            process.wait()
