ARDUINO_LIBRARIES := Arduino_TensorFlowLite

# This example does not need to use a file system
DISABLE_SPIFFS = 1
# Disabling WIFI will give us additional RAM and CPU. 
DISABLE_WIFI = 1
# This is the default baudrate for tensor flow logging 
COM_SPEED_SERIAL = 9600
