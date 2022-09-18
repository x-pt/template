# BUILDING
FROM rockylinux:9 AS builder
LABEL author="Mystic"
WORKDIR /app

# build deps
RUN dnf upgrade --refresh -y && \
    dnf install -y \
        cmake \
        gcc \
        gcc-c++ \
        make && \
    dnf clean all

COPY . .

WORKDIR ./build
RUN cmake .. && cmake --build . --target {{project-name}} --config Release --parallel 8

# DEPLOYING
FROM scratch

WORKDIR /

COPY --from=builder /app/build/{{project-name}} ./

ENTRYPOINT ["/{{project-name}}"]
