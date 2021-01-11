FROM jjanzic/docker-python3-opencv:contrib-opencv-4.0.1
WORKDIR /app
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY config.json /app
COPY ./recognition /app/recognition
CMD ["python", "recognition/main.py"]
