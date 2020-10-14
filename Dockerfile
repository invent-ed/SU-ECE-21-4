FROM jjanzic/docker-python3-opencv
WORKDIR /app
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY ./recognition /app/recognition
WORKDIR /app/recognition
CMD ["python", "recognition.py"]
