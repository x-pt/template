# BUILDING
FROM rockylinux:9 AS builder
LABEL author="{{cookiecutter.full_name}}"
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

COPY .. .

RUN cmake -B build && cmake --build build --target {{cookiecutter.project_slug}} --config Release --parallel 8

# DEPLOYING
{% if cookiecutter.cxx_project_type == "binary" and not cookiecutter.cxx_share_enabled -%}
FROM scratch
{%- else -%}
FROM rockylinux:9-minimal
{%- endif %}

COPY --from=builder /app/build/{{cookiecutter.project_slug}} /{{cookiecutter.project_slug}}

CMD ["/{{cookiecutter.project_slug}}"]
