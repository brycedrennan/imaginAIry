FROM python:3.10.6-slim  as base

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 make

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore


FROM base as build_wheel

RUN pip install wheel

WORKDIR /app

COPY imaginairy ./imaginairy
COPY setup.py README.md ./

RUN python setup.py bdist_wheel



FROM base as install_wheel

WORKDIR /app

COPY requirements-dev.in ./

RUN pip install -r requirements-dev.in

COPY --from=build_wheel /app/dist/* ./

RUN pip install *.whl
RUN imagine --help
COPY Makefile ./
COPY tests ./tests

