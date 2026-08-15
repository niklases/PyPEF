FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code after pip install
COPY run.py /app/
COPY pypef/ /app/pypef/

RUN ["python", "-c", "import torch;print(torch.__version__)"]

EXPOSE 5000

# Not defining entrypoint herein for eased chaining of multiple commands 
# with /bin/bash -c "command1 && command2..."
#ENTRYPOINT ["python", "/app/run.py"]
