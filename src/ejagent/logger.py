import logging


def setup_logger(
    name: str = "ejagent",
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Create and configure a console logger.

    Args:
        name: Logger name.
        level: Minimum log level.
        fmt: Log format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "ejagent") -> logging.Logger:
    """Return a configured logger by name.

    Args:
        name: Logger name.

    Returns:
        Configured logging.Logger instance.
    """
    return setup_logger(name)
