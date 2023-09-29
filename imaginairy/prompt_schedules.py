import csv
import re
from copy import copy

from imaginairy import ImaginePrompt
from imaginairy.utils import frange


def parse_schedule_str(schedule_str):
    """Parse a schedule string into a list of values."""
    pattern = re.compile(r"([a-zA-Z0-9_-]+)\[([a-zA-Z0-9_:,. -]+)\]")
    match = pattern.match(schedule_str)
    if not match:
        msg = f"Invalid kwarg schedule: {schedule_str}"
        raise ValueError(msg)

    arg_name = match.group(1).replace("-", "_")
    if not hasattr(ImaginePrompt(), arg_name):
        msg = f"Invalid kwarg schedule. Not a valid argument name: {arg_name}"
        raise ValueError(msg)

    arg_values = match.group(2)
    if ":" in arg_values:
        start, end, step = arg_values.split(":")
        arg_values = list(frange(float(start), float(end), float(step)))
    else:
        arg_values = parse_csv_line(arg_values)
    return arg_name, arg_values


def parse_schedule_strs(schedule_strs):
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


def prompt_mutator(prompt, schedules):
    """
    Given a prompt and a list of kwarg schedules, return a series of prompts that follow the schedule.

    kwarg_schedules example:
    {
        "prompt_strength": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    """
    schedule_length = len(next(iter(schedules.values())))
    for i in range(schedule_length):
        new_prompt = copy(prompt)
        for attr_name, schedule in schedules.items():
            setattr(new_prompt, attr_name, schedule[i])
        new_prompt.validate()
        yield new_prompt


def parse_csv_line(line):
    reader = csv.reader([line])
    for row in reader:
        parsed_row = []
        for value in row:
            try:
                parsed_row.append(float(value))
            except ValueError:
                parsed_row.append(value)
        return parsed_row
