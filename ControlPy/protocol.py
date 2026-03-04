"""
Message protocol: [4-byte length][1-byte type length][type][payload]
All length fields are big-endian. Payload is UTF-8.
"""

import struct
from typing import Tuple


def encode_message(msg_type: str, payload: bytes) -> bytes:
    """Encode a message with length and type prefix."""
    type_bytes = msg_type.encode("utf-8")
    if len(type_bytes) > 255:
        raise ValueError("Message type must be at most 255 bytes")
    body = bytes([len(type_bytes)]) + type_bytes + payload
    length = len(body)
    return struct.pack(">I", length) + body


def decode_message(data: bytes) -> Tuple[str, bytes]:
    """Decode length-prefixed message; returns (type, payload)."""
    if len(data) < 4:
        raise ValueError("Message too short for length header")
    length, = struct.unpack(">I", data[:4])
    if len(data) < 4 + length:
        raise ValueError("Incomplete message")
    body = data[4 : 4 + length]
    type_len = body[0]
    msg_type = body[1 : 1 + type_len].decode("utf-8")
    payload = body[1 + type_len :]
    return msg_type, payload


def read_message(sock) -> Tuple[str, bytes]:
    """Read one length-prefixed message from socket. Blocks until full message received."""
    header = _read_exact(sock, 4)
    length, = struct.unpack(">I", header)
    body = _read_exact(sock, length)
    type_len = body[0]
    msg_type = body[1 : 1 + type_len].decode("utf-8")
    payload = body[1 + type_len :]
    return msg_type, payload


def _read_exact(sock, n: int) -> bytes:
    buf = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.append(chunk)
        n -= len(chunk)
    return b"".join(buf)
