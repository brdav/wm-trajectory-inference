# Dockerfile for VGGT-SLAM 2.0 inference
# Build with:  podman build -t vggtslam_image .
#
# Target hardware: NVIDIA GH200 (Grace Hopper, linux/aarch64, sm_90). nvcr.io/nvidia/pytorch
# images are multi-arch and ship CUDA/cuDNN/torch already built and tuned for this hardware,
# so no conda is needed to get a CUDA-enabled torch on aarch64 (a plain `pip install torch`
# on aarch64 is CPU-only - PyPI has no CUDA-enabled aarch64 torch wheel).
#
# open3d is the only remaining wrinkle: gtsam-develop (vggt_slam/solver.py's SL(4) pose-graph
# backend) only ships aarch64 wheels from Python 3.11 onward, while open3d's aarch64 wheels on
# PyPI stop at Python 3.11 (0.18.0; 0.19.0 dropped aarch64 entirely) - so Python 3.11 is the
# only version where both have a ready-made wheel. No nvcr.io/nvidia/pytorch tag ships Python
# 3.11 though (the series jumps straight from 3.10 to 3.12), so on this image's Python 3.12,
# open3d has to be built from source (stage 1 below) while gtsam-develop still installs from a
# plain pip wheel (it has a cp312/aarch64 one).

# ---- Stage 1: build an open3d wheel for cp312/aarch64 (no such wheel exists on PyPI) ----
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS open3d-builder
ENV DEBIAN_FRONTEND=noninteractive
ARG OPEN3D_REF=v0.19.0

# Same C++ build deps as Open3D's own CI (util/install_deps_ubuntu.sh), minus the
# clang/ninja/libsdl2-dev/libc++-dev/libxi-dev filament-only ones - BUILD_GUI=OFF below skips
# compiling Filament (the GUI/visualization backend) entirely, and this pipeline never uses
# open3d's visualization module (only PointCloud/voxel_down_sample/write_point_cloud). GLFW is
# still built as a third-party dep even with BUILD_GUI=OFF though, and on this Ubuntu 24.04
# base its CMake configure also probes for a Wayland backend - pkg-config/libwayland-dev/
# libwayland-bin/wayland-protocols/libxkbcommon-dev satisfy that (install_deps_ubuntu.sh
# doesn't need these since Open3D's own CI images are Ubuntu 20.04, predating GLFW's Wayland
# support probe).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        xorg-dev \
        libxcb-shm0 \
        libglu1-mesa-dev \
        libtbb-dev \
        libosmesa6-dev \
        libudev-dev \
        autoconf \
        libtool \
        gfortran \
        pkg-config \
        libwayland-dev \
        libwayland-bin \
        wayland-protocols \
        libxkbcommon-dev \
    && rm -rf /var/lib/apt/lists/*

# cmake from pip rather than apt: guarantees CMake >=3.24 (Open3D's minimum) regardless of the
# base image's Ubuntu release, without tracking an apt package version for it.
RUN pip install --no-cache-dir cmake

RUN git clone --depth 1 --branch ${OPEN3D_REF} https://github.com/isl-org/Open3D.git /opt/open3d-src
WORKDIR /opt/open3d-src

# BUILD_SHARED_LIBS/CUDA/PYTORCH_OPS/TENSORFLOW_OPS/SYCL=OFF matches Open3D's own arm64 release
# wheel config (docker/docker_build.sh's openblas_export_env()). BUILD_GUI=OFF is our own
# addition on top of that (see comment above) - this is a headless compute node, so there's no
# working display for the GUI to use even if it were built.
#
# Deliberately NOT -GNinja: Open3D's own build (docker/Dockerfile.ci) uses the default Unix
# Makefiles generator, and its OpenBLAS third-party build is wired as a Makefiles-style
# ExternalProject step - under Ninja it fails with "'openblas/lib/libopenblas.a' ... missing
# and no known rule to make it" because that step's output isn't declared as a Ninja byproduct.
RUN cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_GUI=OFF \
        -DBUILD_PYTORCH_OPS=OFF \
        -DBUILD_TENSORFLOW_OPS=OFF \
        -DBUILD_SYCL_MODULE=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -B build \
    && cmake --build build --parallel "$(nproc)" --target pip-package
# `pip-package` (as opposed to `install-pip-package`) only builds the wheel into
# build/lib/python_package/pip_package/, without installing it into this stage - stage 2 below
# copies just that wheel out, so none of this stage's build toolchain/object files/Open3D
# source tree end up in the final image.

# ---- Stage 2: runtime image ----
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# GH200 is sm_90; keeps any third-party CUDA-extension builds from wasting time compiling for
# other architectures (and from autodetecting a GPU that may not be visible at build time).
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"

# git          - to fetch the DINOv2 backbone via torch.hub at runtime (salad) and to clone
#                map-matching-2 below.
# wget/unzip   - used by this repo's own download helpers.
# ffmpeg       - to extract frames from raw video clips.
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1 - runtime deps of opencv-python/open3d.
# cmake/make/g++/python3-dev/python-is-python3 - build deps for map-matching-2 below (its own
# CMake fetches and builds every other dependency -- Boost, zlib, libosmium, etc. -- from source,
# so nothing else is needed at the apt level).
# osmium-tool - the `osmium` CLI (extract / tags-filter), used by scripts/osm_maps/osm_bev.py's `prepare` to pre-filter
# country PBFs; conda-forge has no aarch64 build of it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        unzip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        cmake make g++ python3-dev python-is-python3 \
        osmium-tool \
    && rm -rf /var/lib/apt/lists/*

COPY --from=open3d-builder /opt/open3d-src/build/lib/python_package/pip_package/open3d-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/open3d-*.whl && rm /tmp/open3d-*.whl

# Install this repo's dependencies (+ package metadata, so `pip install -e . --no-build-
# isolation` against the live, volume-mounted checkout at runtime is near-instant - every
# dependency is already satisfied) from a throwaway copy of the current source tree, then
# discard that copy. The image intentionally does NOT keep a copy of the repo itself: the real
# code always comes from a bind mount at runtime, so this image never goes stale relative to it.
COPY . /opt/vggtslam-src
RUN cd /opt/vggtslam-src \
    # torch/torchvision come from this base image already (CUDA-matched to it); reinstalling
    # them from requirements.txt would silently replace them with a non-CUDA aarch64 build.
    # open3d comes from the from-source wheel installed above, not PyPI (see top comment).
    # opencv-python is stripped here too -- replaced below by a from-source, CUDA-enabled build
    # (cudaoptflow/cudaimgproc) so vggt_slam/frame_overlap.py's per-frame keyframe-selection
    # tracking can run on the GPU's dedicated optical-flow hardware instead of the CPU; pip's
    # opencv-python wheel is CPU-only (no cv2.cuda backend at all, confirmed via
    # cv2.cuda.getCudaEnabledDeviceCount() returning 0 despite the module being importable).
    && sed -i '/^torch==/d;/^torchvision==/d;/^open3d/d;/^opencv-python/d' requirements.txt \
    # Skip Perception Encoder / SAM 3 (setup.sh steps 4-5): they're only used for the optional
    # --run_os open-set object detection feature, and pull in decord==0.6.0, which has no
    # linux-aarch64 build anywhere (not even a source path) - not worth blocking core SLAM
    # inference on this hardware.
    && sed -i '/^# 4\. Install Perception Encoder/,/^# 6\. Install current repo/{/^# 6\. Install current repo/!d}' setup.sh \
    # setup.sh installs this repo and its third_party forks (salad, vggt) editable, pointing at
    # this throwaway copy - since that copy is deleted below, install everything as a real,
    # non-editable copy instead. The user's own `pip install -e .` at runtime (against the live
    # mount) switches just this repo's own package back to editable, near-instantly.
    && sed -i 's/pip install -e /pip install /g' setup.sh \
    && chmod +x setup.sh && ./setup.sh \
    && cd / && rm -rf /opt/vggtslam-src

# setuptools/wheel only exist transiently inside pip's isolated build env during the `pip
# install .` above - not a runtime dependency, so it doesn't persist in site-packages. Install
# them for real here so a later `pip install -e . --no-build-isolation` against the live,
# volume-mounted checkout doesn't need network access (compute nodes may not have any).
RUN pip install --no-cache-dir setuptools wheel

# Extra packages for plotting/analyzing trajectories (geospatial plotting)
RUN pip install --no-cache-dir pyproj shapely

# osmium (vggt_slam/map_match.py's local OSM-region-extract cropping, GPS fusion): baked in here
# instead of pip-installed at the start of every batch_infer.py run -- compute nodes on this
# cluster hit intermittent pypi.org DNS resolution failures ("Name or service not known") that
# made that runtime install unreliable, and it added real startup latency to every single job.
RUN pip install --no-cache-dir osmium

# CUDA-enabled OpenCV (replaces the CPU-only opencv-python wheel stripped from requirements.txt above) so
# vggt_slam/frame_overlap.py's keyframe-selection tracking can run on the GPU's optical-flow hardware
# (rationale and profiling: memory/performance-investigations.md).
#
# Built in-place (no separate wheel artifact the way open3d's stage produces one -- OpenCV's own
# CMake install target places the compiled cv2 extension directly into PYTHON3_PACKAGES_PATH) and
# the source/build tree is removed afterward, matching the map-matching-2 build's own pattern
# below. BUILD_LIST is deliberately NOT used to trim modules: calib3d (vggt_slam/undistort.py's
# initUndistortRectifyMap) transitively needs features2d/flann, and video (calcOpticalFlowPyrLK,
# still used as a fallback/reference) is small enough that trimming it isn't worth the risk of
# missing a transitive dependency. GUI backends (WITH_GTK/WITH_QT=OFF) are skipped instead --
# nothing in this pipeline calls cv2.imshow outside the debug-only visualize=True path in
# frame_overlap.py, and this is a headless compute node with no display anyway.
ARG OPENCV_REF=4.10.0
RUN git clone --depth 1 --branch ${OPENCV_REF} https://github.com/opencv/opencv.git /opt/opencv-src \
    && git clone --depth 1 --branch ${OPENCV_REF} https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib-src \
    && cd /opt/opencv-src \
    # Default (Unix Makefiles) generator, not -GNinja: ninja isn't pip-installed until the
    # map-matching-2 build below, and apt's cmake here is already new enough for OpenCV (unlike
    # map-matching-2, which specifically needs cmake>=4.2).
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-src/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=9.0 \
        -DCUDA_ARCH_PTX=9.0 \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_CUDNN=OFF \
        -DOPENCV_DNN_CUDA=OFF \
        -DBUILD_opencv_cudaoptflow=ON \
        -DBUILD_opencv_cudaimgproc=ON \
        -DBUILD_opencv_cudaarithm=ON \
        -DBUILD_opencv_cudawarping=ON \
        -DBUILD_opencv_cudafilters=ON \
        -DBUILD_opencv_python3=ON \
        -DBUILD_opencv_python2=OFF \
        -DPYTHON3_EXECUTABLE=$(which python3) \
        -DPYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
        -DOPENCV_ENABLE_NONFREE=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -B build \
    && cmake --build build --parallel "$(nproc)" --target install \
    && ldconfig \
    && cd / && rm -rf /opt/opencv-src /opt/opencv_contrib-src

# map-matching-2 (https://github.com/addy90/map-matching-2): snaps a noisy/predicted GPS track
# onto real OSM road centerlines before/while it's used as a SLAM fusion prior (see
# vggt_slam/map_match.py). CLI-only (no Python bindings, no pip package) -- vggt_slam/map_match.py
# shells out to the installed `map_matching_2` binary. Its CMake requires >=4.2 (this base image's
# apt cmake is 3.22, hence the pip-installed one below) and C++20 std::format, which needs GCC
# 13+ (this base image's g++ is 11.4, which fails with "'format' is not a member of 'std'"); this
# base image (nvcr.io/nvidia/pytorch) ships no conda at all, so a throwaway Miniforge install
# (conda-forge's own minimal installer, self-contained under /tmp) supplies that GCC just for
# this build -- the resulting binary links only against libc/libm (confirmed via ldd; its own
# CMake sets "static linking of system libraries: ON"), so it needs nothing from that env at
# runtime, and the whole build tree (including Miniforge and the from-source Boost/etc. build
# products fetched via CMake FetchContent, multiple GB) is removed below, leaving only the
# ~40MB installed binary.
RUN pip install --no-cache-dir "cmake>=4.2" ninja \
    && wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /tmp/miniforge3 \
    && rm /tmp/miniforge.sh \
    && /tmp/miniforge3/bin/conda create -n mm2build -c conda-forge gxx=13 gcc=13 -y -q \
    && git clone --depth 1 https://github.com/addy90/map-matching-2.git /opt/map-matching-2 \
    && cd /opt/map-matching-2 \
    && CC=/tmp/miniforge3/envs/mm2build/bin/gcc CXX=/tmp/miniforge3/envs/mm2build/bin/g++ \
       cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build \
    && cmake --build build --parallel "$(nproc)" --target install \
    && install -Dm755 run/gcc/release/bin/map_matching_2 /usr/local/bin/map_matching_2 \
    && cd / && rm -rf /opt/map-matching-2 /tmp/miniforge3

CMD ["/bin/bash"]
