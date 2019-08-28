#!/bin/bash

set -e

rm -rf venv && python -m venv venv && source venv/bin/activate

rm -rf libyaml \
  && git clone https://github.com/yaml/libyaml libyaml && pushd libyaml \
  && ./bootstrap && ./configure --prefix=$HOME/libyaml \
  && make && make install \
  && rm -rf PyYAML-5.1 && wget http://pyyaml.org/download/pyyaml/PyYAML-5.1.tar.gz \
  && tar xvf PyYAML-5.1.tar.gz && pushd PyYAML-5.1 \
  && sed -i '/#include_dirs/s/#include.*/include_dirs=$HOME\/libyaml\/include/' setup.cfg \
  && sed -i '/#library_dirs/s/#library.*/library_dirs=$HOME\/libyaml\/lib/' setup.cfg \
  && python setup.py --with-libyaml -I $HOME/libyaml/include -L $HOME/libyaml/lib install \
  && popd && popd \
  && rm -rf PyYAML-5.1.tar.gz PyYAML-5.1 libyaml

pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt

rm -rf apex \
  && git clone https://github.com/NVIDIA/apex && pushd apex \
  && sed -i '/.*check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/s/^/#/' setup.py \
  && pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
  && popd \
  && rm -rf apex
