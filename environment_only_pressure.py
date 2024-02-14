import gym
import numpy as np
from gym import spaces
import serial
import serial.tools.list_ports
import DeviceUserProtocolHandle as deviceUser
import time
from queue import Queue
from threading import Thread, Lock, Event
from DeviceUserExample import packetHandleThread

uartFifo = Queue(200)
cmdAckEvent = Event()
uartPacektFifo = deviceUser.ProtocolDatasFIFO()


class SmartBedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, port='COM3', baudrate=115200):
        self.alpha = 0.1
        self.episode = 1
        self.output_episode = 100
        self.time_per_action = 0.1

        self.action_space = spaces.MultiDiscrete([4] * 6)  # 0:no change, 1:inflation, 2:stop, 3:deflation
        low_obs = np.full(16, 0).astype(np.float32)
        high_obs = np.full(16, 1).astype(np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs)
        self.obs = np.zeros(16)
        self.previous_pressure_values = np.zeros(16)
        self.previous_action = np.zeros(6)  # here need to change to the previous inner pressure of the airbag

        # Serial port initialization for communication with the smart bed hardware
        self.cmdLock = Lock()
        self.get_uart_ports()
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.inter_byte_timeout = 0.01
        self.ser.timeout = 2
        self.open_serial_port()

        self.uartReceiveThread = Thread(target=self.uart_receive_task)
        self.uartReceiveThread.start()
        self.packetParaseThread = packetHandleThread()
        self.packetParaseThread.airPressReflash.connect(self.airbag_pressure_display)
        self.packetParaseThread.start()

    def get_uart_ports(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("Error finding any port!")
            return None
        for port in ports:
            print(port.description)

    def open_serial_port(self):
        try:
            self.ser.open()
            print("Serial port opened successfully.")
        except Exception as e:
            print(f"Error opening serial port: {e}")

    def close_serial_port(self):
        try:
            self.ser.close()
            print("Serial port closed successfully.")
        except Exception as e:
            print(f"Error closing serial port: {e}")

    def uart_receive_task(self):
        while True:
            if self.ser.is_open:
                try:
                    data = self.ser.read(10000)
                except:
                    continue
                t = time.time()
                print("UartReceiveTask rec data", int(round(t * 1000)), data)
                if len(data):
                    uartFifo.put(data)
                    uartPacektFifo.enqueue(data)
            else:
                time.sleep(0.1)

    def cmd_packet_exec(self, cmdPacket):
        self.cmdLock.acquire()
        if self.ser.is_open:
            print("ser.write:", cmdPacket.hex())
            try:
                self.ser.write(cmdPacket)
            except:
                print("ser.write fail")
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

    def airbag_pressure_display(self, data):
        for i in range(6):
            print("index[%d]: %.2f" % (i, data[i]))

    def send_airbag_control_command(self, action):
        index = 0
        cfgTime = 0XFF  # 1-20(S) or 0XFF(always run)
        mapByte = deviceUser.airMap()
        mapByte.char = 0  # resets all airbag controls to 0

        # Update mapByte based on the action for each airbag
        airbag_mapping = ['xiaoTui', 'daTui', 'Tun', 'Yao', 'Xiong', 'Jian']  # according to map_bits
        for i, act in enumerate(action):
            if act > 0:
                setattr(mapByte.bit, airbag_mapping[i], 1)  # eg. mapByte.bit.xiaoTui = 1

        if mapByte.char == 0:
            print("No airbag control action specified.")
            return

        # action_code: 1: inflation, 2: stop, 3: deflation
        if 1 in action:
            action_code = 1
        elif 2 in action:
            action_code = 2
        elif 3 in action:
            action_code = 3
        else:
            action_code = 0

        # Convert the action to the corresponding command packet
        cmdPacketData = deviceUser.airControlCmdPacketSend(index, action_code, mapByte.char, cfgTime)
        if not self.cmd_packet_exec(cmdPacketData):
            print("Cmd Error: No response!")
        else:
            print("Cmd success!")

    def read_pressure_data(self):
        # read map pressure
        if self.ser.in_waiting:
            rawBytesArr = self.ser.read(self.ser.in_waiting)
            lowpresDataList = deviceUser.lowPressAnalysis(rawBytesArr)

            if lowpresDataList is not None:
                print(lowpresDataList)
                return lowpresDataList
            else:
                print("Error processing pressure data.")
                return None
        else:
            print("No data available from serial port.")
            return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_reward = 0.0
        self.action_time = 0.0
        self.action_time_steps = 0
        self.obs = np.zeros(16)  # pressure
        self._get_obs = self.obs.astype(np.float32)
        print('train_obs: ', self._get_obs)
        return self._get_obs, {}

    def step(self, action):
        reward = 0.0
        done = False
        self.action_time_steps += 1
        self.action_time += self.time_per_action

        # action: 0:no change, 1:inflation, 2:stop, 3:deflation
        # take action of 6 airbags

        pressure_values = self.read_pressure_data()
        if pressure_values is not None:
            self.obs = pressure_values  # Update pressure in observation

        if self.obs == 0:
            done = True

        pressure_variance = np.var(pressure_values)

        pressure_change_continuity = np.mean(np.abs(self.previous_pressure_values - pressure_values))
        self.previous_pressure_values = pressure_values.copy()

        action_change_continuity = np.mean(np.abs(self.previous_action - action))
        self.previous_action = action.copy()

        # set reward
        # pressure distribution
        if pressure_variance == 0:
            reward += 10.0
        else:
            reward += 1.0 / pressure_variance

        reward -= pressure_change_continuity
        reward -= action_change_continuity

        if done:
            self.episode += 1

        return self._get_obs, reward, done, False, {}

    def __del__(self):
        self.close_serial_port()
