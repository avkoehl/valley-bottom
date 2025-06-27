def format_time_duration(seconds):
    """
    Format seconds into a human-readable time string.
    For longer durations, shows hours and minutes; for shorter ones, shows minutes and seconds.
    """

    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds:.2f}s"
