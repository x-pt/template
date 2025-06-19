# {{cookiecutter.package_name}}/app.py

import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Configure basic logging
# This will output logs to the console with a timestamp, logger name, log level, and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get a logger for this module (e.g., "my_package_name.app")
logger = logging.getLogger(__name__)

def run_server(port: int = 8000, server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler) -> None:
    """
    Starts a simple HTTP server on the specified port.

    Args:
        port: The port number on which to run the server. Defaults to 8000.
        server_class: The server class to use. Defaults to HTTPServer.
        handler_class: The request handler class to use. Defaults to SimpleHTTPRequestHandler,
                       which serves files from the current directory.
    """
    server_address = ('', port)
    # The httpd object is an instance of server_class.
    # It will handle incoming HTTP requests using an instance of handler_class.
    httpd = server_class(server_address, handler_class)

    logger.info(f"Starting HTTP server on http://localhost:{port}/")
    logger.info(f"Serving files from the current directory using {handler_class.__name__}.")
    logger.info("Press Ctrl+C to stop the server.")

    try:
        # serve_forever() handles requests until an explicit shutdown() or an exception occurs.
        httpd.serve_forever()
    except KeyboardInterrupt:
        # This block handles a clean shutdown when Ctrl+C (SIGINT) is pressed.
        logger.info("Keyboard interrupt received, shutting down the server...")
        httpd.shutdown() # Gracefully shut down the HTTP server.
        logger.info("Server shut down successfully.")
    except Exception as e:
        # This block handles any other unexpected exceptions during server operation.
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log stack trace
        httpd.shutdown() # Ensure server is shut down on other errors too.
        logger.info("Server shut down due to an error.")

if __name__ == "__main__":
    # This block ensures that run_server() is called only when the script is executed directly,
    # for example, by running `python -m {{cookiecutter.package_name}}.app`.
    # It prevents run_server() from being called if this module is imported by another script.
    run_server()
