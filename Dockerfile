FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Preload and store PLMs, increases image size by ~ 10 GBs, comment for skipping caching
# e.g., for mounting your own cache directories or downloading weights every new docker run
RUN python -c "from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer; \
    AutoModelForMaskedLM.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True); \
    AutoTokenizer.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True); \
    AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    AutoModelForMaskedLM.from_pretrained('facebook/esm1v_t33_650M_UR90S_3'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_3')"

# Copy application code after pip install
COPY run.py /app/
COPY pypef/ /app/pypef/

RUN ["python", "-c", "import torch;print(torch.__version__)"]

EXPOSE 5000

# Not defining entrypoint herein for eased chaining of multiple commands 
# with /bin/bash -c "command1 && command2..."
#ENTRYPOINT ["python", "/app/run.py"]
