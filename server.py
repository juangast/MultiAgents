"""
Daniel - A01199140
"""

import socket
import json
import threading
import math

HOST = "127.0.0.1"
PORT = 5000


def get_state(step):
    x = 2.0 + math.sin(step * 0.1) * 3.0
    return {
        "step": step,
        "agents": [
            {"id": 1, "x": x, "y": 0, "z": 3, "rotation": 0}
        ]
    }


def handle_client(conn, addr):
    print(f"Cliente conectado: {addr}")
    step = 0
    buffer = ""
    with conn:
        while True:
            try:
                data = conn.recv(1024)
            except ConnectionResetError:
                break

            if not data:
                break

            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if line == "GET_STATE":
                    step += 1
                    response = json.dumps(get_state(step)) + "\n"
                    conn.sendall(response.encode("utf-8"))

    print(f"Cliente desconectado: {addr}")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Servidor escuchando en {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        thread.start()


if __name__ == "__main__":
    main()
