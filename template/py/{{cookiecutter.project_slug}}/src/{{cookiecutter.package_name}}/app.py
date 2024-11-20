from http.server import HTTPServer, SimpleHTTPRequestHandler


def run_server(port: int = 8000) -> None:
    handler = SimpleHTTPRequestHandler
    with HTTPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server...")
            httpd.shutdown()


if __name__ == "__main__":
    run_server()
