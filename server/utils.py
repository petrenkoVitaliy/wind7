from datetime import datetime


def tprint(*args, **kwargs):
    """Prints a message with the current timestamp."""
    timestamp = datetime.now().strftime("[%M:%S.%f]")
    print(timestamp, *args, **kwargs)
