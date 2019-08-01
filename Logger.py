import logging

logging.basicConfig(level=logging.INFO,
                    format='%(module)s:%(lineno)d [%(threadName)s] [%(levelname)-5.5s]  %(message)s',
                    handlers=[logging.FileHandler('logger.log', mode='a'),
                              logging.StreamHandler()])

logger = logging.getLogger('logger')


def setLevel(level):
    global logger
    if level == 'debug':
        level = logging.DEBUG
    elif level == 'info':
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)
