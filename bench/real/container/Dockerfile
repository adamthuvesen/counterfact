FROM ghcr.io/astral-sh/uv@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca
RUN uv pip install --system --no-cache pytest==8.3.5 iniconfig==2.1.0 packaging==24.2 pluggy==1.5.0
COPY verify.py /opt/counterfact/verify.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LC_ALL=C.UTF-8 TZ=UTC HOME=/workspace
USER 65534:65534
WORKDIR /workspace
ENTRYPOINT ["python", "-I", "/opt/counterfact/verify.py"]
