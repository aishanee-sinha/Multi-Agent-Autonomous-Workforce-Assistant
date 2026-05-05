FROM public.ecr.aws/lambda/python:3.11
RUN yum install -y gcc gcc-c++ gzip tar make
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY src/ ${LAMBDA_TASK_ROOT}/
RUN find ${LAMBDA_TASK_ROOT} -name "*.py" -exec chmod 755 {} \; && \
    find ${LAMBDA_TASK_ROOT} -type d -exec chmod 755 {} \; && \
    chown -R root:root ${LAMBDA_TASK_ROOT}
ENV HOME="/tmp"
CMD ["orchestrator.sqs_handler"]