# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

FROM continuumio/miniconda3 as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the requirements and environment
COPY requirements.txt . \
     environment.yaml .

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    conda env create -f environment.yaml 

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "auto_qchem", "/bin/bash", "-c"]

# Install XTB and openbabel 
RUN apt-get update && apt-get install -y \
    openbabel

RUN conda install --channel conda-forge openbabel xtb-python && conda clean -a -y

RUN pip install git+https://github.com/dkesada/auto-qchem_exp.git && pip cache purge

# Copy the source code into the container.
COPY . .


ENV KWARGS="{\"user_id\": \"f6a7ec39-9358-4015-8615-bb421eaf4cea\", \"smiles\": [\"CC1(C)OB([Cu][P](OC(C)C)(C2=CC=CC=C2)OC(C)C)OC1(C)C\", \"CC1(OB(OC1(C)C)[Cu][P](OC)(OC)C2=CC=CC=C2)C\"]}"
CMD ["conda", "run", "--no-capture-output", "-n", "auto_qchem", "python3", "scripts/main.py"]
