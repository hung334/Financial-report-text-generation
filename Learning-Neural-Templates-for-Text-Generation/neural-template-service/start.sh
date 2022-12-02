#!/bin/sh
LOGS_DIR=logs

if [ ! -d "${LOGS_DIR}" ]
then
  mkdir "${LOGS_DIR}"
fi

python2 neural-template-service.py neural-template-service.conf
