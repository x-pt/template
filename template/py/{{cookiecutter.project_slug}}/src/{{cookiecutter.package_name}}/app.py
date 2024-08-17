from http.server import HTTPServer, SimpleHTTPRequestHandler


def run_server(port=8000):
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
