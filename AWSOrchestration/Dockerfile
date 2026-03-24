# Use the official AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.11

# 1. Install system tools needed for potential C-extensions
RUN yum install -y gcc gcc-c++ gzip tar make

# 2. Install the Rust compiler (Keep this for tiktoken/tokenization support)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Copy and install requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Use a cache mount to speed up re-builds
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all source files
COPY state.py           ${LAMBDA_TASK_ROOT}/
COPY slack_agent.py     ${LAMBDA_TASK_ROOT}/
COPY calendar_agent.py  ${LAMBDA_TASK_ROOT}/
COPY orchestrator.py    ${LAMBDA_TASK_ROOT}/

# Lambda handler entrypoint
CMD ["orchestrator.handler"]
