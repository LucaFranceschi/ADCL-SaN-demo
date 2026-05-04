FROM continuumio/miniconda3:25.3.1-1

ENV HOME=/home/user
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR $HOME/app

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY environment.yaml $HOME/app/environment.yaml

# Run conda/pip as root so it can write to /opt/conda
RUN conda env create -f $HOME/app/environment.yaml \
    && conda clean -a -y

ENV PATH=/opt/conda/envs/acl-ssl2/bin:$PATH
# SHELL ["/bin/bash", "-lc"]

COPY app-requirements.txt $HOME/app/app-requirements.txt

RUN conda run -n acl-ssl2 pip install --no-cache-dir --upgrade -r $HOME/app/app-requirements.txt

RUN useradd -m -u 1000 user

COPY --exclude=data . $HOME/app
RUN chown -R user:user /home/user/
RUN mkdir -p /tmp/gradio && chown -R user:user /tmp/gradio

USER user

CMD ["python", "app.py"]