import binascii
import ctypes

c_uint8 = ctypes.c_uint8
c_uint16 = ctypes.c_uint16
c_uint32 = ctypes.c_uint32


class deviceData(ctypes.BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Temperature", c_uint16),
        ("AirPress", c_uint32),
    ]


class map_bit(ctypes.BigEndianStructure):
    _fields_ = [
        ("XiaoTui", c_uint8, 1),
        ("DaTui", c_uint8, 1),
        ("Tun", c_uint8, 1),
        ("Yao", c_uint8, 1),
        ("Xiong", c_uint8, 1),
        ("Jian", c_uint8, 1),
    ]


class airPressInfoCmdParam(ctypes.BigEndianStructure):
    _fields_ = [
        ("map", c_uint16),
        ("data", deviceData * 12),
    ]


class lowPressInfoCmdParam(ctypes.BigEndianStructure):
    _fields_ = [
        ("index", c_uint8),
        ("data", c_uint16 * 16),
    ]


class map_bits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("xiaoTui", c_uint8, 1),
        ("daTui", c_uint8, 1),
        ("Tun", c_uint8, 1),
        ("Yao", c_uint8, 1),
        ("Xiong", c_uint8, 1),
        ("Jian", c_uint8, 1),
    ]


class airMap(ctypes.Union):
    _fields_ = [("bit", map_bits),
                ("char", c_uint8)]


# 数据处理类
class ProtocolDatasFIFO:
    def __init__(self):
        self.data = bytearray()

    def enqueue(self, data):
        self.data.extend(data)

    def dequeue(self):
        return self.data.pop(0)

    def peek(self, index):
        if 0 <= index < len(self.data):
            return self.data[index]
        return None

    def clear(self):
        self.data.clear()

    def get_fifo_length(self):
        return len(self.data)

    def collect_protocol_packet(self, p_Packet):
        # 检查输入参数
        if p_Packet is None:
            print("Error: Invalid input parameter p_Packet is None")
            return -1

        # 协议包解析
        start_byte = None
        index = 0
        while self.get_fifo_length():
            # 从FIFO中读取字节，直到找到起始字节0xA5
            data = self.peek(index)
            if data == 0xA5:
                print("Found start byte:", hex(data))
                index += 1
                break
            else:
                self.dequeue()
                if self.get_fifo_length() == 0:
                    print("Not found start byte")
                    return -1

        # 检查协议包长度
        packet_len = self.peek(index)
        if packet_len is None:
            print("Warning: Unable to read packet length from FIFO")
            return -1

        if packet_len < 5:
            print("Warning: Invalid packet length:", packet_len)
            self.dequeue()
            return -2

        if packet_len > 255:
            print("Warning: Invalid packet length:", packet_len)
            self.dequeue()
            return -3

        # 检查协议包尾部
        fifoLength = self.get_fifo_length()
        if packet_len > fifoLength:
            print("The current packet is incomplete")
            return -4

        last_byte = self.peek(packet_len - 1)
        if last_byte != 0x55:
            print("Warning: Invalid end byte:", hex(last_byte))
            self.dequeue()
            return -5
        # 检查协议包CRC

        # 从FIFO中读取协议包字段
        for i in range(3):
            p_Packet.append(self.dequeue())

        # 读取剩余的数据
        if packet_len > 5:
            param_buff_len = packet_len - 5
            for i in range(param_buff_len):
                p_Packet.append(self.dequeue())
        p_Packet.append(self.dequeue())
        p_Packet.append(self.dequeue())
        # 输出调试信息
        print("Successfully collected protocol packet:", p_Packet.hex())
        return 0


# 校验数据生成
def CheckSum(data, len):
    crcNum = 0
    if len:
        for i in range(len):
            crcNum += data[i]
    checkSum = crcNum & 0XFF
    return checkSum


# 数据包生成
def PacketGeneration(cmd, data, dataLen):
    sendData = bytearray(dataLen + 5)

    sendData[0] = 0xA5
    sendData[1] = dataLen + 5
    sendData[2] = cmd
    if dataLen:
        for i in range(dataLen):
            sendData[3 + i] = data[i]

    sendData[3 + dataLen] = CheckSum(sendData, dataLen + 3)
    sendData[4 + dataLen] = 0X55
    return sendData


'''
index:0 右侧气囊 1:左侧气囊
action: 1:充气 2:暂停 3:放气
map: 0-5 bit分别对应小腿、大腿、臀、腰、胸、肩
time:1-20S 或0XFF(一直充放)
'''


# 气囊充放气控制
def airControlCmdPacketSend(index, action, map, time):
    data = bytearray(4)
    data[0] = index
    data[1] = action
    data[2] = map
    data[3] = time
    return PacketGeneration(0x02, data, len(data))


# 气压连续监测控制
def airPressCtlPacketSend(interval, map):
    data = bytearray(4)
    data[0] = (interval & 0XFFFF) >> 8
    data[1] = (interval & 0XFF)
    data[2] = (map & 0XFFFF) >> 8
    data[3] = (map & 0XFF)
    return PacketGeneration(0x05, data, len(data))


# 充气模式控制
def airModeCtlPacketSend(mode):
    data = bytearray(1)
    data[0] = mode
    return PacketGeneration(0x08, data, len(data))


# 电机升降控制
def motorCtlPacketSend(index, action):
    data = bytearray(2)
    data[0] = index
    data[1] = action
    return PacketGeneration(0x09, data, len(data))


# 温度控制
def temperCtlPacketSend(index, action, targetTemp):
    data = bytearray(3)
    data[0] = index
    data[1] = action
    data[2] = targetTemp
    return PacketGeneration(0x0A, data, len(data))


# 温度读取
def temperRequirePacketSend(index):
    data = bytearray(1)
    data[0] = index
    return PacketGeneration(0x0B, data, len(data))


# 气压计算
def calculate_press(raw_data):
    f_dat = 0.0
    press_data = 0.0
    dat = int(raw_data)
    if dat > 8388608:
        f_dat = (dat - 16777216) / 8388608.0
    else:
        f_dat = dat / 8388608.0

    press_data = round((250 * f_dat + 25) * 1000, 2)
    return press_data


# 低精度压力反馈
def lowPressAnalysis(rawBytesArr):
    if len(rawBytesArr) is not (1 + (2 * 16)):
        print("lowPressAnalysis param err", len(rawBytesArr))
        return None
    cmdParamData = lowPressInfoCmdParam()
    raw_data = (ctypes.c_char * len(rawBytesArr)).from_buffer(rawBytesArr)
    ctypes.memmove(ctypes.byref(cmdParamData), raw_data, ctypes.sizeof(cmdParamData))
    hex_data = binascii.hexlify(ctypes.string_at(ctypes.byref(cmdParamData), ctypes.sizeof(cmdParamData)))
    print(hex_data)
    lowpresDataList = [{} for _ in range(16)]
    for i in range(16):
        lowpresDataList[i] = cmdParamData.data[i]
    print(lowpresDataList)
    return lowpresDataList


# 气压反馈解析
'''
map(2B) 0-5bit表示右侧气囊组气压,8-14bit表示左侧气囊组气压
'''


def airPressAnalysis(rawBytesArr):
    if len(rawBytesArr) is not 2 + (12 * 6):
        print("airPressAnalysis param err", len(rawBytesArr))
        return False
    cmdParamData = airPressInfoCmdParam()
    raw_data = (ctypes.c_char * len(rawBytesArr)).from_buffer(rawBytesArr)
    ctypes.memmove(ctypes.byref(cmdParamData), raw_data, ctypes.sizeof(cmdParamData))

    hex_data = binascii.hexlify(ctypes.string_at(ctypes.byref(cmdParamData), ctypes.sizeof(cmdParamData)))
    print(hex_data)
    presDataList = [{} for _ in range(12)]

    print("map 0x%x" % cmdParamData.map)
    for i in range(12):
        print("temp = 0x%x,press = 0x%x" % (cmdParamData.data[i].Temperature, cmdParamData.data[i].AirPress))

    # 右侧气囊气压计算
    for i in range(6):
        if (cmdParamData.map >> i) & 0X01:
            if cmdParamData.data[i].AirPress:
                cuePressData = calculate_press(cmdParamData.data[i].AirPress)
                print("right index:", i, "press:", cuePressData)
                presDataList[i] = cuePressData
            else:
                presDataList[i] = 0.0
        else:
            presDataList[i] = 0.0
    # 左侧气囊气压计算
    for i in range(6):
        if (cmdParamData.map >> (i + 8)) & 0X01:
            if cmdParamData.data[i + 6].AirPress is not 0:
                cuePressData = calculate_press(cmdParamData.data[i + 6].AirPress)
                print("left index:", i, "press:", cuePressData)
                presDataList[i + 6] = cuePressData
            else:
                presDataList[i + 6] = 0.0
        else:
            presDataList[i + 6] = 0.0
    print(presDataList)
    return presDataList
