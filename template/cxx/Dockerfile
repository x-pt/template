# BUILDING
FROM rockylinux:9 AS builder
LABEL author="{{authors}}"
WORKDIR /app

# build deps
RUN dnf upgrade --refresh -y && \
    dnf --enablerepo=crb install -y \
        cmake \
        gcc \
        gcc-c++ \
        make \
        # enable repo crb to install the following packages for building static binary
        libstdc++-static \
        glibc-static && \
    # due to this Dockerfile is a multi-stage, the following clean steps are redundant.
    dnf clean all && \
    rm -fr /var/cache/yum

COPY . .

WORKDIR ./build
RUN cmake .. && cmake --build . --target {{project-name}} --config Release --parallel 8

# DEPLOYING
FROM scratch

COPY --from=builder /app/build/{{project-name}} /{{project-name}}

CMD ["/{{project-name}}"]
