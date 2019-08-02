#!/bin/bash

set -e

rm -rf venv && python -m venv venv && source venv/bin/activate

pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl

rm -rf /tmp/libyaml \
  && git clone https://github.com/yaml/libyaml /tmp/libyaml \
  && pushd /tmp/libyaml \
  && ./bootstrap \
  && ./configure \
  && make \
  && sudo make install \
  && wget http://pyyaml.org/download/pyyaml/PyYAML-5.1.tar.gz \
  && tar xvf PyYAML-5.1.tar.gz \
  && cd PyYAML-5.1 \
  && python setup.py --with-libyaml install \
  && popd

pip install -r requirements.txt