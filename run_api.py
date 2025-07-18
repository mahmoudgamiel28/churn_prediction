"""
Script to run the FastAPI application with dynamic port allocation
"""

import uvicorn
import socket
import os
from src.api.main import app

def find_available_port(start_port=8000, max_attempts=10):
    """
    Find an available port starting from start_port
    
    Args:
        start_port (int): Starting port number
        max_attempts (int): Maximum number of ports to try
        
    Returns:
        int: Available port number
        
    Raises:
        RuntimeError: If no available port is found
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")

if __name__ == "__main__":
    # Get initial port from environment variable or use default
    initial_port = int(os.getenv("API_PORT", 8000))
    
    try:
        # Find available port
        available_port = find_available_port(initial_port)
        
        if available_port != initial_port:
            print(f"⚠️  Port {initial_port} is not available. Using port {available_port} instead.")
        else:
            print(f"✅ Starting API server on port {available_port}")
            
        print(f"🚀 API will be available at: http://localhost:{available_port}")
        print(f"📚 API Documentation: http://localhost:{available_port}/docs")
        print(f"💊 Health Check: http://localhost:{available_port}/health")
        
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=available_port,
            reload=True,
            log_level="info"
        )
        
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        exit(1)