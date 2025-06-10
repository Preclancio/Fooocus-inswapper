import os
from IPython.utils import capture

with capture.capture_output() as cap:
    if not os.path.exists('/content/sd/libtcmalloc/libtcmalloc_minimal.so.4'):
        %env CXXFLAGS=-std=c++14
        !mkdir -p /content/temporal
        %cd /content/temporal
        !wget -q {giturl1} && tar zxf gperftools-2.5.tar.gz && mv gperftools-2.5 gperftools
        !wget -q {giturl2}
        %cd /content/temporal/gperftools
        !patch -p1 < /content/temporal/Patch
        !./configure --enable-minimal --enable-libunwind --enable-frame-pointers --enable-dynamic-sized-delete-support --enable-sized-delete --enable-emergency-malloc; make -j4
        !mkdir -p /content/sd/libtcmalloc && cp .libs/libtcmalloc*.so* /content/sd/libtcmalloc
        %env LD_PRELOAD=/content/sd/libtcmalloc/libtcmalloc_minimal.so.4
        %cd /content/temporal
        !rm *.tar.gz Patch && rm -r /content/temporal
    else:
        !wget -N -c /content/sd/libtcmalloc/libtcmalloc_minimal.so.4 'https://drive.google.com/uc?export=download&id=1RN7lMuorcZEhV17jBwdux9s9GjCInSEP'
        %env LD_PRELOAD=/content/sd/libtcmalloc/libtcmalloc_minimal.so.4
