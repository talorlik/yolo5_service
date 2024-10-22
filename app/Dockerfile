FROM ultralytics/yolov5:latest-cpu
# Install cron
RUN apt-get update && apt-get install -y cron
# Copy the cleanup script
COPY prediction_cleanup.sh /usr/local/bin/prediction_cleanup.sh
# Make the script executable
RUN chmod +x /usr/local/bin/prediction_cleanup.sh
# Add the cron job
RUN (crontab -l 2>/dev/null; echo "*/1 * * * * /usr/local/bin/prediction_cleanup.sh") | crontab -

WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app"

# Start cron and the application using bash
CMD ["bash", "-c", "service cron start && python3 app.py"]
