class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def disconnect(self, callback):
        """Remove a callback from the signal."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def disconnect_all(self):
        """Remove all callbacks."""
        self.callbacks.clear()

    def emit(self, *args, **kwargs):
        for cb in self.callbacks[:]:
            cb(*args, **kwargs)