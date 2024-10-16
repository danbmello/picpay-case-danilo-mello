FROM python:3.12-slim

# Accept user-related arguments from the build context
ARG USER_ID
ARG USER_NAME
ARG MODEL_TYPE

# Set user-related environment variables

ENV HOME=/home/${USER_NAME} \
    VIRTUAL_ENV=/home/${USER_NAME}/venv

ENV \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    BUILD_POETRY_LOCK=${HOME}/poetry.lock.build \
    SPARK_HOME=/opt/spark \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 
    
ENV \
    PYTHONPATH=${SPARK_HOME}/python:$PYTHONPATH \
    PATH=${JAVA_HOME}/bin:${VIRTUAL_ENV}/bin:${SPARK_HOME}/python:/usr/local/bin:$PATH

# Install dependencies
RUN apt-get -qq update \
    && apt-get -qq -y install vim curl build-essential libb64-dev python3-venv default-jre scala \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

# Create user and group
RUN addgroup --system --gid ${USER_ID} ${USER_NAME} \
    && useradd --system -m --no-log-init --home-dir ${HOME} --uid ${USER_ID} --gid ${USER_NAME} --groups ${USER_NAME} ${USER_NAME}

RUN chown -R ${USER_NAME}:${USER_NAME} ${HOME}
RUN mkdir -p /app \
    && chown -R ${USER_NAME}:${USER_NAME} /app

USER root

# Install Spark
RUN curl -fsSL https://downloads.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz | tar -xz -C /opt/ \
    && mv /opt/spark-3.5.3-bin-hadoop3 ${SPARK_HOME} \
    && pip install pyspark

RUN chown -R ${USER_NAME}:${USER_NAME} ${SPARK_HOME} \
    && chmod 777 ${SPARK_HOME} \
    && chmod 777 ${SPARK_HOME}/python \
    && chmod 777 ${SPARK_HOME}/python/pyspark

# Install pip and setuptools globally as root to avoid permission issues
RUN pip install --upgrade pip setuptools

# Install Poetry globally as root to ensure it can be accessed system-wide
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1 \
    && ln -s /home/${USER_NAME}/.local/bin/poetry /usr/local/bin/poetry

# Ensure Poetry is accessible globally and fix permissions for Poetry config
RUN mkdir -p /home/${USER_NAME}/.config/pypoetry \
    && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.config/pypoetry \
    && chmod -R 777 /home/${USER_NAME}/.config/pypoetry

USER ${USER_NAME}

# Copy project files
COPY pyproject.toml *.lock /app/
WORKDIR /app

RUN python3.12 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools \
    && ${VIRTUAL_ENV}/bin/pip install poetry \
    && ${VIRTUAL_ENV}/bin/poetry install --only main \
    && cp poetry.lock ${BUILD_POETRY_LOCK} \
    && rm -rf ${HOME}/.cache/*

# Copy application code
COPY . /app/

# Expose the application's port
EXPOSE 8081

# Start the container
#CMD ["tail", "-f", "/dev/null"]
CMD ["python", "./src/main.py"]


