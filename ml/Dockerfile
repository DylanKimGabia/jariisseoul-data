# 예측 API 컨테이너 예시. 저장소 루트에서 빌드:  docker build -f ml/Dockerfile -t jariml .
# 실행:  docker run -p 8000:8000 -v $(pwd)/data:/app/data jariml
#   data 폴더를 연결하지 않으면 시작할 때 GitHub에서 수집 데이터를 받는다 (수백 MB).
#   운영에서는 api/app.py의 recent()를 DB 조회로 바꾸면 data 폴더가 필요 없다.
FROM python:3.11-slim
WORKDIR /app/ml
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml .
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
