# Importing Libraries 
import serial 
import time 


arduino = serial.Serial(port='usb:14300000', baudrate=115200, timeout=.1) 
"""
void setInputMessage(void) {
  inputs[0] = myserial_control_in.pos_x;
  inputs[1] = myserial_control_in.pos_y;
  inputs[2] = myserial_control_in.pos_z;
  inputs[3] = myserial_control_in.vel_body_x;
  inputs[4] = myserial_control_in.vel_body_y;
  inputs[5] = myserial_control_in.vel_body_z;
  inputs[6] = myserial_control_in.roll;
  inputs[7] = myserial_control_in.pitch;
  inputs[8] = myserial_control_in.yaw;
  inputs[9] = myserial_control_in.gyro_x;
  inputs[10] = myserial_control_in.gyro_y;
  inputs[11] = myserial_control_in.gyro_z;

  // inputs = [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, roll_target, pitch_target];
  // DEBUG_serial.printf("%f, %f, %f, %f, %f, %f, %f, %f\n", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
  set_network_input(&controller, inputs);
}

void setOutputMessage(void) {
  myserial_control_out.motor_1 = saturateSignedInt16(controller.out[0]);
  myserial_control_out.motor_2 = saturateSignedInt16(controller.out[1]);
  myserial_control_out.motor_3 = saturateSignedInt16(controller.out[2]);
  myserial_control_out.motor_4 = saturateSignedInt16(controller.out[3]);
}"""
# Function to write and read data from Arduino
def write_read(x): 
    # use conventions from input message and output message
    # x = "1,2,3,4,5,6,7,8,9,10,11"
    x = x.encode()
    arduino.write(x)
    time.sleep(0.005)
    data = arduino.readline()
    return data
    