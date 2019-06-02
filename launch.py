import threading
import socket
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler
import time
import signal
import urls
import json
import os

INITIAL_PORT_VALUE = (
    8000
)  # The http server will try to open on port 7860. If not available, 7861, 7862, etc.
TRY_NUM_PORTS = (
    100
)  # Number of ports to try before giving up and throwing an exception.


def get_first_available_port(initial, final):
    """
    Gets the first open port in a specified range of port numbers
    :param initial: the initial value in the range of port numbers
    :param final: final (exclusive) value in the range of port numbers, should be greater than `initial`
    :return:
    """
    for port in range(initial, final):
        try:
            s = socket.socket()  # create a socket object
            s.bind(("localhost", port))  # Bind to the port
            s.close()
            return port
        except OSError:
            pass
    raise OSError(
        "All ports from {} to {} are in use. Please close a port.".format(
            initial, final
        )
    )


def serve_files_in_background(port, directory_to_serve=None):
    class HTTPHandler(SimpleHTTPRequestHandler):
        """This handler uses server.base_path instead of always using os.getcwd()"""

        def _set_headers(self):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

        def log_message(self, format, *args):
            return

        def do_GET(self):
            path = os.path.join(self.path, '')  # adds trailing '/' if none is there
            if path in urls.get_endpoints:
                self._set_headers()
                msg = urls.get_endpoints[path]()
                self.wfile.write(json.dumps(msg).encode('utf-8'))
            else:
                super().do_GET()

        def do_POST(self):
            path = os.path.join(self.path, '')  # adds trailing '/' if none is there
            if path in urls.post_endpoints:
                self._set_headers()
                print("works")
                msg = json.loads(self.rfile.read())
                urls.post_endpoints[path](msg)
            else:
                pass
                # TODO(aliabd): raise exception per best practice

    class HTTPServer(BaseHTTPServer):
        """The main server, you pass in base_path which is the path you want to serve requests from"""

        def __init__(self, base_path, server_address, RequestHandlerClass=HTTPHandler):
            self.base_path = base_path
            BaseHTTPServer.__init__(self, server_address, RequestHandlerClass)

    httpd = HTTPServer(directory_to_serve, ("localhost", port))

    # Now loop forever
    def serve_forever():
        # try:
        while True:
            # sys.stdout.flush()
            httpd.serve_forever()
        # except (KeyboardInterrupt, OSError):
        #     httpd.server_close()

    thread = threading.Thread(target=serve_forever, daemon=True)
    thread.start()

    return httpd


def start_simple_server(directory_to_serve=None):
    port = get_first_available_port(
        INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS
    )
    print("SERVING AT PORT:", port)
    httpd = serve_files_in_background(port, directory_to_serve)
    return port, httpd


def close_server(server):
    server.shutdown()
    server.server_close()


port, httpd = start_simple_server()


class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass


def service_shutdown(signum, frame):
    print('Shutting server down due to signal %d' % signum)
    httpd.shutdown()
    raise ServiceExit


signal.signal(signal.SIGTERM, service_shutdown)
signal.signal(signal.SIGINT, service_shutdown)

try:
    # Keep the main thread running, otherwise signals are ignored.
    while True:
        time.sleep(0.5)
except ServiceExit:
    pass

