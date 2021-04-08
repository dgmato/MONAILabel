from abc import abstractmethod


class MONAIApp(object):
    def __init__(self, app_dir, **kwargs):
        self.app_dir = app_dir

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def infer(self, request):
        pass

    @abstractmethod
    def train(self, request):
        pass

    @abstractmethod
    def next_sample(self, request):
        pass

    @abstractmethod
    def save_label(self, request):
        pass
