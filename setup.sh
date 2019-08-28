#!/bin/bash

set -e

rm -rf venv && python -m venv venv && source venv/bin/activate

rm -rf libyaml \
  && git clone https://github.com/yaml/libyaml libyaml && pushd libyaml \
  && ./bootstrap && ./configure --prefix=$HOME/libyaml \
  && make && make install \
  && rm -rf PyYAML-5.1 && wget http://pyyaml.org/download/pyyaml/PyYAML-5.1.tar.gz \
  && tar xvf PyYAML-5.1.tar.gz && pushd PyYAML-5.1 \
  && python setup.py build_ext --include-dirs=$HOME/libyaml/include --library-dirs=$HOME/libyaml/lib \
  && python setup.py --with-libyaml install \
  && popd && popd \
  && rm -rf PyYAML-5.1.tar.gz PyYAML-5.1 libyaml

python -m pip install -r requirements.txt

rm -rf apex \
  && git clone https://github.com/NVIDIA/apex && pushd apex \
  && sed -i '/.*check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/s/^/#/' setup.py \
  && python -m pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
  && popd \
  && rm -rf apex
