## 1. System dependency 

```bash
sudo apt-get update
sudo apt-get install -y \
    libatlas-base-dev \
    libportaudio2 \
    libopenjp2-7 \
    libtiff6 \
    libopenexr-dev \
    libgfortran5 \
    portaudio19-dev \
    python3-pip \
    python3-venv
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install neccessary libraries

```bash
pip install --upgrade pip
pip install opencv-python-headless \
            edge_impulse_linux \
            scipy \
            numpy \
            psutil \
            pyaudio \
            six
```