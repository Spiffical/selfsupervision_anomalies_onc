from main import create_app
from config import ARGS
import os
import socket

if __name__ == '__main__':
    app = create_app(ARGS)
    
    # Resolve host/port with basic conflict handling
    def find_free_port(preferred: int = 8050) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(('127.0.0.1', preferred)) != 0:
                return preferred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    host = os.environ.get("HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("PORT", "8050"))
    port = find_free_port(preferred_port)
    if port != preferred_port:
        print(f"Port {preferred_port} in use, switching to {port}")
    app.run(debug=False, host=host, port=port)
