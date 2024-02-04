'''
Author: YourName
Date: 2024-01-19 09:30:37
LastEditTime: 2024-01-19 10:35:00
LastEditors: YourName
Description: 
FilePath: \simulation_model (2)\DeviceUserExample.py
版权声明
'''

import DeviceUserProtocolHandle
from DeviceUserProtocolHandle import *

from collections import deque
import time
from queue import Queue
from threading import Thread, Lock, Event
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QThread, pyqtSignal, QCoreApplication
import serial
import serial.tools.list_ports

uartFifo = Queue(200)
cmdAckEvent = Event()
uartPacektFifo = ProtocolDatasFIFO()


class packetHandleThread(QThread):
    airPressReflash = pyqtSignal(list)

    def __init__(self):
        super().__init__()

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            data = uartFifo.get()
            print("packetHandleTask  packet data:", data.hex())
            if len(data) == 0:
                continue
            packet = bytearray()
            while uartPacektFifo.get_fifo_length():
                packet.clear()
                if uartPacektFifo.collect_protocol_packet(packet) == 0:
                    if packet[0] == 0XA5 and packet[len(packet) - 1] == 0X55 and len(packet) == packet[1]:
                        print("get a vaild packet data")
                        cmd = packet[2]
                        packetLength = packet[1]
                        if (
                                cmd == 0x02 or cmd == 0x03 or cmd == 0x04 or cmd == 0x05 or cmd == 0x08 or cmd == 0x09 or cmd == 0x0A or cmd == 0X0D):
                            print("cmdAckEvent.set(),cmd", cmd)
                            cmdAckEvent.set()
                        elif cmd == 0x06:  # 压力数据通知
                            pass
                        elif cmd == 0x07:  # 气压数据通知
                            if len(packet) - 5 != 12 * 6 + 2:
                                print("invaild packet")
                                continue
                            rawDataArr = bytearray(12 * 6 + 2)
                            rawDataArr[0:(12 * 6 + 2)] = packet[3:len(packet) - 2]
                            presDataList = DeviceUserProtocolHandle.airPressAnalysis(rawDataArr)
                            self.airPressReflash.emit(presDataList)
                        elif cmd == 0X0B:  # 温度数据
                            curTemper = packet[6]
                            print("cmdAckEvent.set(),cmd", cmd)
                            cmdAckEvent.set()


class UserExample():
    def __init__(self, parent=None):
        # super().__init__(parent)
        self.cmdLock = Lock()
        self.get_uart_ports()
        self.ser = serial.Serial()
        self.uartReceiveThread = Thread(target=self.uartReceiveTask)
        self.uartReceiveThread.start()
        self.packetParaseThread = packetHandleThread()
        self.packetParaseThread.airPressReflash.connect(self.airDisplay)
        self.packetParaseThread.start()

    def get_uart_ports(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("未找到可用串口")
            return None
        for port in ports:
            print(port.description)

    # uart接收数据线程处理
    def uartReceiveTask(self):
        while True:
            if self.ser.is_open:
                try:
                    data = self.ser.read(10000)
                except:
                    continue
                t = time.time()
                print("uartReceiveTask rec data", int(round(t * 1000)), data)
                if len(data):
                    uartFifo.put(data)
                    uartPacektFifo.enqueue(data)
            else:
                time.sleep(0.1)

    def cmdPacketExec(self, cmdPacket):
        self.cmdLock.acquire()
        if self.ser.is_open:
            print("self.ser.write:", cmdPacket.hex())
            try:
                self.ser.write(cmdPacket)
            except:
                print("self.ser.write fail")
                # QMessageBox.critical(self, "Cmd Error", "write fail")
                self.cmdLock.release()
                return False
            if cmdAckEvent.wait(1.0):
                cmdAckEvent.clear()
                self.cmdLock.release()
                print("cmd ack ok")
                return True
            else:
                print("cmd ack timeout")
        self.cmdLock.release()

    # 打开串口
    def port_open(self):
        self.ser.port = 'COM3'
        self.ser.baudrate = 115200
        self.ser.inter_byte_timeout = 0.01
        self.ser.timeout = 2
        try:
            self.ser.open()
        except:
            print("Port Error", "此串口不能被打开！")
            return None
        print("串口打开成功", self.ser.portstr)

    # 关闭串口
    def port_close(self):
        try:
            self.ser.close()
        except:
            pass
        print("串口关闭成功", self.ser.portstr)

    # 发送数据
    def data_send(self, data):
        if self.ser.is_open:
            num = self.ser.write(data)
        else:
            pass
        return False

    def airDisplay(self, data):
        # 气压数据依据下标0-5顺序为 小腿、大腿、臀、腰、胸、肩
        for i in range(6):
            print("index[%d]: %.2f" % (i, data[i]))


if __name__ == "__main__":
    userExample = UserExample()
    userExample.port_open()
    if userExample.ser.is_open:

        '''
        #气囊控制
        index  =  0    #0:右侧 1：左侧
        action =  2    #1充气 2停止 3放气
        cfgTime = 0XFF #1-20(S) 或0XFF(一直执行)
        mapByte = airMap()
        mapByte.bit.Jian    = 1
        mapByte.bit.Xiong   = 1
        mapByte.bit.Yao     = 1
        mapByte.bit.Tun     = 1
        mapByte.bit.daTui   = 1
        mapByte.bit.xiaoTui = 1
        print(str(mapByte.char))
        if(mapByte.char == 0):#需要操作的位置需要置1，如果都为0即没有需要操作的位置，该参数无效
            print("请选择需要控制的气囊") 
        else:          
            cmdPacketData = DeviceUserProtocolHandle.airControlCmdPacketSend(index,action,mapByte.char,cfgTime)
            if userExample.cmdPacketExec(cmdPacketData) == False:
                print("Cmd Error", "指令无应答")
            else:
                print("Cmd success")                 
        userExample.port_close()               
        '''

        # 气压监测控制：
        intervalTime = 500  # 监测间隔（500 - 5000）
        checkMap = 0XFFFF
        cmdPacketData = DeviceUserProtocolHandle.airPressCtlPacketSend(intervalTime, checkMap)
        if not userExample.cmdPacketExec(cmdPacketData):
            print("Cmd Error", "指令无应答")
        else:
            print("Cmd success")
            # userExample.port_close()
