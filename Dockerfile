FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 gcc

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN pip install imaginairy
RUN imagine --help
