import serial
import serial.tools.list_ports
from threading import Thread
import time
from queue import Queue
from UserProtocolHandle import ProtocolDatasFIFO, collect_raw_packet
from single_line_data_for_real_time_withoutAvg import LowPressureData2img, load_model
import pytz
from datetime import datetime

pressDataList = []


class userUartReceive:
    def __init__(self):
        self.ser = serial.Serial()
        self.uartReceiveThread = Thread(target=self.uartReceiveTask)

    def port_open(self):
        self.ser.port = 'COM4'
        self.ser.baudrate = 115200
        self.ser.inter_byte_timeout = 0.01
        self.ser.timeout = 2
        try:
            self.ser.open()
        except:
            print("此串口不能被打开")
            return None

    def uartReceiveTask(self):
        global pressDataList
        while True:
            if self.ser.is_open:
                try:
                    data = self.ser.read(10000)
                except:
                    continue

                if len(data):
                    t = time.time()
                    # print("uartReceiveTask rec data", int(round(t * 1000)), data)
                    packet = bytearray()
                    cData = bytearray(data)
                    timestamp = datetime.now().strftime("[%H:%M:%S.%f]")[:-4] + "]"
                    while len(cData):
                        packet.clear()
                        collect_raw_packet(cData, packet)
                        curData = timestamp + packet.hex().upper()

                        if len(packet) == 35:
                            pressDataList.append(curData)
                            # if len(pressDataList) < 10:
                            #     pressDataList.append(curData)
                            # elif len(pressDataList) == 10:
                            #     pressDataList[0:9] = pressDataList[1:-1]
                            #     pressDataList[-1] = curData
                        elif len(packet) == 17:
                            LowPressureData2img(load_model(), pressDataList[-1], curData)
                            pressDataList = []

            else:
                time.sleep(0.2)
                print("此串口未被打开")


if __name__ == "__main__":
    userUart = userUartReceive()
    userUart.port_open()
    userUart.uartReceiveThread.start()
