import logging
import logging.handlers
"""
\033[0;30m   dark
\033[0;33m  brown
\033[0;36m  cyan 青色
\033[0;37m  灰色gray
"""


class Logger(object):
    def __init__(self,log_name,log_level):
        # self.log_name = log_name
        # self.log_level = log_level
        # self.logger = self.log_init()
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        self.sh = logging.StreamHandler()


    def log_init(self):
        logger_name = self.log_name
        # log_file = self.config.get("log", "log_file")
        logger_level = self.log_level

        logger = logging.getLogger(logger_name)
        formatter =logging.Formatter("%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s")

        # fileHandler = logging.handlers.RotatingFileHandler(log_file,maxBytes=8388608888, backupCount=5, encoding='utf-8')
        # fileHandler.setFormatter(formatter)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(logger_level)

        logger.addHandler(streamHandler)

        return logger

    def debug(self, message):
        # \033[0;32m  green
        self.fontColor('\033[0;32m%s\033[0m')
        self.logger.debug(message)

    def info(self, message):
        # \033[0;34m 蓝色 \033[1;34m light_blue
        self.fontColor('\033[0;34m%s')
        self.logger.info(message)

    def warning(self, message):
        self.fontColor('\033[0;37m%s\033[0m')
        self.logger.warning(message)

    def error(self, message):
        # \033[0;31m  red
        self.fontColor('\033[0;31m%s\033[0m')
        self.logger.error(message)

    def cri(self, message):
        # \033[0;35m  purple
        self.fontColor('\033[0;35m%s\033[0m')
        self.logger.critical(message)

    def fontColor(self, color):
        formatter = logging.Formatter(color % '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
        self.sh.setFormatter(formatter)
        self.logger.addHandler(self.sh)