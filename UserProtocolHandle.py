

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

def collect_raw_packet(rawdata,p_Packet):

    # 检查输入参数
    if p_Packet is None:
        print("Error: Invalid input parameter p_Packet is None")
        return -1

    # 协议包解析
    start_byte = None
    index = 0
    while (len(rawdata)):
        # 从FIFO中读取字节，直到找到起始字节0xA5
        data = rawdata[index]
        if data == 0xAA or data == 0xAB:
            print("Found start byte:", hex(data))
            index += 1
            start_byte = data
            break
        else:
            rawdata.pop(0)
            if(len(rawdata) == 0):
                print("Not found start byte")
                return -1

    # 检查协议包长度
    packet_len = rawdata[index]
    if packet_len is None:
        print("Warning: Unable to read packet length from FIFO")
        return -1

    if ((packet_len == 35 and start_byte == 0XAA) or (packet_len == 17 and start_byte == 0xAB)) != 1:  
        print("Warning: Invalid packet length:", packet_len)
        rawdata.pop(0)
        return -2
    
    # 检查协议包尾部
    fifoLength = len(rawdata)
    if packet_len > fifoLength:
        print("The current packet is incomplete")
        return -4

    last_byte = rawdata[packet_len - 1]
    if last_byte != 0x55:
        print("Warning: Invalid end byte:", hex(last_byte))
        rawdata.pop(0)
        return -5
    # 检查协议包CRC

    # 从FIFO中读取协议包字段
        # 从FIFO中读取协议包字段
    if(start_byte == 0XAA):
        for i in range(35):
            p_Packet.append(rawdata.pop(0))
    elif(start_byte == 0XAB):
        for i in range(17):
            p_Packet.append(rawdata.pop(0))
    # 输出调试信息
    print("collect_raw_packet Successfully collected protocol packet:", p_Packet.hex())


    return 0     
