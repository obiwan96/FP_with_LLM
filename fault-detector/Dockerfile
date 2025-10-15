FROM python:3.11-slim

RUN apt update && apt install -y --no-install-recommends \
    iproute2 curl iputils-ping net-tools \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m obiwan
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY fault_detector.py .

# 실행 유저 전환
USER obiwan

# 기본 실행 커맨드
CMD ["python", "fault_detector.py"]
