# Use a slim version of Python 3.12 as the base image
FROM python:3.12-slim

# Accept user-related arguments from the build context
ARG USER_ID
ARG USER_NAME
ARG MODEL_TYPE

# Set user-related environment variables
ENV HOME=/home/${USER_NAME} \
    VIRTUAL_ENV=/home/${USER_NAME}/venv

# Set various environment variables to configure Python, Spark, and Java
ENV \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    BUILD_POETRY_LOCK=${HOME}/poetry.lock.build \
    SPARK_HOME=/opt/spark \  S
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Extend the PATH and PYTHONPATH to include Spark and Java binaries
ENV \
    PYTHONPATH=${SPARK_HOME}/python:$PYTHONPATH \
    PATH=${JAVA_HOME}/bin:${VIRTUAL_ENV}/bin:${SPARK_HOME}/python:/usr/local/bin:$PATH

# Update apt-get and install required dependencies
RUN apt-get -qq update \
    && apt-get -qq -y install vim curl build-essential libb64-dev python3-venv default-jre scala \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

# Create user and group based on passed arguments
RUN addgroup --system --gid ${USER_ID} ${USER_NAME} \
    && useradd --system -m --no-log-init --home-dir ${HOME} --uid ${USER_ID} --gid ${USER_NAME} --groups ${USER_NAME} ${USER_NAME}

# Set the correct ownership for the user's home directory
RUN chown -R ${USER_NAME}:${USER_NAME} ${HOME}

# Create an /app directory and set ownership
RUN mkdir -p /app \
    && chown -R ${USER_NAME}:${USER_NAME} /app

# Switch to root to install additional software
USER root

# Download and install Spark, then install PySpark via pip
RUN curl -fsSL https://downloads.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz | tar -xz -C /opt/ \
    && mv /opt/spark-3.5.3-bin-hadoop3 ${SPARK_HOME} \  
    && pip install pyspark

# Adjust permissions for Spark and PySpark directories to avoid access issues
RUN chown -R ${USER_NAME}:${USER_NAME} ${SPARK_HOME} \
    && chmod 777 ${SPARK_HOME} \
    && chmod 777 ${SPARK_HOME}/python \
    && chmod 777 ${SPARK_HOME}/python/pyspark

# Upgrade pip and setuptools globally to avoid permission issues
RUN pip install --upgrade pip setuptools

# Install Poetry globally for managing Python dependencies
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1 \
    && ln -s /home/${USER_NAME}/.local/bin/poetry /usr/local/bin/poetry

# Ensure Poetry is accessible globally and fix permissions for Poetry config
RUN mkdir -p /home/${USER_NAME}/.config/pypoetry \
    && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.config/pypoetry \
    && chmod -R 777 /home/${USER_NAME}/.config/pypoetry

# Switch back to the created user
USER ${USER_NAME}

# Copy the pyproject.toml and lock files to /app
COPY pyproject.toml *.lock /app/

# Set /app as the working directory
WORKDIR /app

# Create a Python virtual environment, upgrade pip, and install Poetry dependencies
RUN python3.12 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools \  
    && ${VIRTUAL_ENV}/bin/pip install poetry \  
    && ${VIRTUAL_ENV}/bin/poetry install --only main \  
    && cp poetry.lock ${BUILD_POETRY_LOCK} \  
    && rm -rf ${HOME}/.cache/*  

# Expose port 8080 for the application
EXPOSE 8080

# Set the default command to run the application
CMD ["python", "./src/main.py"]

