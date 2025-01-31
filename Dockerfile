FROM continuumio/miniconda3

# Install gcc and other build tools
RUN apt-get update && \
    apt-get install -y gcc build-essential

WORKDIR /app

RUN conda create -n ml_env python=3.11.5 && \
    conda create -n subq_env python=3.11.5


SHELL ["/bin/bash", "-c"]
RUN echo "source activate ml_env" > ~/.bashrc
ENV PATH /opt/conda/envs/ml_env/bin:$PATH

RUN conda run -n ml_env conda install -c conda-forge keras==2.12.0 tensorflow==2.12.0 pandas==2.0.3 numpy==1.25.0 -y
RUN conda run -n ml_env conda install -c conda-forge biopython -y
RUN conda run -n ml_env conda install -c bioconda hmmer -y 
RUN conda run -n ml_env conda install bioconda::anarci 

RUN conda run -n subq_env conda install -c conda-forge numpy==2.1.2 pandas==2.2.3 -y

COPY . /app

COPY requirements.txt /app/requirements.txt
RUN conda run -n ml_env pip install -r requirements.txt
RUN conda run -n subq_env pip install -r requirements.txt

# Run app.py when the container launches
CMD ["gunicorn", "--timeout", "120", "app:app"]