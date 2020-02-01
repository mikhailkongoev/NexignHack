# NexignHack
Face recognition for office pass system within Nexign Hackaton. 01.02.2020



1. Download openvino: https://yadi.sk/d/InVrH4-YwFLI3w

2. Run Apache Kafka (and set host and port in application.properties of event-processing and ws-inegration-service)

3. Run ws-inegration-service

4. Run event-processing service

5. Run face-recognition
5.1 Run 'pip install -r requirements' for dependencies installation
5.2 Run 'python video_service.py'

6. Open 'localhost:9292' for UI