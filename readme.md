# Intelligent Monitoring System Based on Raspberry Pi

this system is divided into two subsystems: background data management system and client system. The background data management system is deployed on PC and the client system is deployed on Raspberry Pi.

## Background Data Management System

- This system is based on Flask framework and python
- The use of database is Redis(for windows)
- The system contains a total of 7 pages and 1 public template
- It can also receive the images captured from the Raspberry Pi in real time. The tool used is Image-zmq, which is specially used to transfer image data between the Raspberry Pi and the PC

## Client System

- This system is based on python, PyQt5, tensorflow and tensorflow lite
- In this system, the images acquired in real time will be recognized and processed
- It contains two core algorithm: face recognition and moving object detection. Respectively using MobileNetV2 and background subtraction method to achieve
- The startup time of the system will be relatively long, maybe need 1 minute

## Operation

- In order to get the project running, you should first run the Flask framework(the code should be `set APP=login.py; set ENV=development; flask run`) and start the Redis server on Windows(remember to turn off the LAN firewall and change the configuration file of Redis, otherwise the Client System will not be allowed to access the Redis database. The code should be `redis-server.exe --service-start`)
- You also need to make sure that the port used to transfer image data is free so that the system can occupy
- After the above preparations are done, you can start running the code file
- the command `python login.py`should be run on the PC
- then the command `python doubleThread.py`should be run on the Raspberry Pi
- There is a detailed explanation in the code