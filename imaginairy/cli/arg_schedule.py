"""Decorator and parsers for command scheduling"""

from typing import Iterable

from imaginairy.utils import frange


def with_arg_schedule(f):
    """Decorator to add arg-schedule functionality to a click command."""

    def new_func(*args, **kwargs):
        arg_schedules = kwargs.pop("arg_schedules", None)

        if arg_schedules:
            schedules = parse_schedule_strs(arg_schedules)
            schedule_length = len(next(iter(schedules.values())))
            for i in range(schedule_length):
                for attr_name, schedule in schedules.items():
                    kwargs[attr_name] = schedule[i]
                f(*args, **kwargs)
        else:
            f(*args, **kwargs)

    return new_func


def parse_schedule_strs(schedule_strs: Iterable[str]) -> dict:
    """Parse and validate input prompt schedules."""
    schedules = {}
    for schedule_str in schedule_strs:
        arg_name, arg_values = parse_schedule_str(schedule_str)
        schedules[arg_name] = arg_values

    # Validate that all schedules have the same length
    schedule_lengths = [len(v) for v in schedules.values()]
    if len(set(schedule_lengths)) > 1:
        raise ValueError("All schedules must have the same length")

    return schedules


def parse_schedule_str(schedule_str):
    """Parse a schedule string into a list of values."""
    import re

    pattern = re.compile(r"([a-zA-Z0-9_-]+)\[([a-zA-Z0-9_:,. -]+)\]")
    match = pattern.match(schedule_str)
    if not match:
        msg = f"Invalid kwarg schedule: {schedule_str}"
        raise ValueError(msg)

    arg_name = match.group(1).replace("-", "_")

    arg_values = match.group(2)
    if ":" in arg_values:
        start, end, step = arg_values.split(":")
        arg_values = list(frange(float(start), float(end), float(step)))
    else:
        arg_values = parse_csv_line(arg_values)
    return arg_name, arg_values


def parse_csv_line(line):
    import csv

    reader = csv.reader([line])
    for row in reader:
        parsed_row = []
        for value in row:
            try:
                parsed_row.append(float(value))
            except ValueError:
                parsed_row.append(value)
        return parsed_row
