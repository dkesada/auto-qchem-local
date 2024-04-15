import logging


logger = logging.getLogger(__name__)


class LogExtractor:
    def __init__(self, log=None):
        if log:
            self.logger = log
        else:
            self.logger = logger
