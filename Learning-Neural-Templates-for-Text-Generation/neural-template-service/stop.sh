#!/bin/sh
SERVICE_NAME="neural-template-service.py"

SERVICE_PID=`ps -ef | grep "${SERVICE_NAME}" | grep -v "grep" | awk '{print $2}'`
if [ ! "${SERVICE_PID}" ]
then
  echo "${SERVICE_NAME} has been stopped!"
  exit 0
fi

for pid in ${SERVICE_PID}
do
  echo "${SERVICE_NAME} pid is ${pid}"
  kill -9 ${pid}  
done
echo "${SERVICE_NAME} stopped!"
