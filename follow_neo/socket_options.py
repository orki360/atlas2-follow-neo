"""Latency options only; never change application acknowledgement semantics."""
import socket


def enable_nodelay(sock):
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return bool(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
    except (OSError, AttributeError):
        # Offline socket pairs and some injected test transports are not TCP.
        return False
