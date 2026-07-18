class NoOpWriter:
    """Writer that preserves the trainer interface without external services."""

    def __init__(self, logger, project_config, **kwargs):
        self.step = 0
        self.mode = ""

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_checkpoint(self, *args, **kwargs):
        pass
