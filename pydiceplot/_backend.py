_backends = ["plotly", "matplotlib"]
_backend = "plotly"


def set_backend(backend: str):
    assert backend in _backends, f"Backend '{backend}' is not available. Choose from {_backends}."
    global _backend
    _backend = backend
