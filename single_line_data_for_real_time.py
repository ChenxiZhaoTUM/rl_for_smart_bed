import re
import math, os
import torch
import numpy as np
from model_CRNN import CRNN
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

torch.set_printoptions(edgeitems=3, linewidth=200, precision=10, sci_mode=False, threshold=5000)


def process_pressure_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AA23") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 64:
                processed_hex_values = ''.join(
                    [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
                     range(0, len(pres_hex_values), 4)])

                pres_decimal_arr = [4095 - int(processed_hex_values[i:i + 4], 16) for i in
                                    range(0, len(processed_hex_values), 4)]

                return time, pres_decimal_arr

    return time, []


def process_sleep_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AB11") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 28:
                processed_hex_values = pres_hex_values[0: 12]

                processed_hex_values = processed_hex_values + ''.join(
                    [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
                     range(12, 20, 4)])

                processed_hex_values = processed_hex_values + pres_hex_values[20: 22]

                processed_hex_values = processed_hex_values + ''.join(
                    [pres_hex_values[24: 26] + pres_hex_values[22: 24]])

                processed_hex_values = processed_hex_values + pres_hex_values[26:]

                pres_decimal_arr = [int(processed_hex_values[i:i + 2], 16) for i in range(0, 12, 2)]
                pres_decimal_arr.extend([int(processed_hex_values[12:16], 16)])
                pres_decimal_arr.extend([int(processed_hex_values[16:20], 16)])
                pres_decimal_arr.append(int(processed_hex_values[20:22], 16))
                pres_decimal_arr.append(int(processed_hex_values[22:26], 16))
                pres_decimal_arr.append(int(processed_hex_values[26:28], 16))

                return time, pres_decimal_arr

    return time, []


def change_input_dimension(input_data_arr, input_sleep_data_arr):
    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = torch.tensor(input_sleep_data_arr[ch])  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data_arr[j])

    return new_input_data


input_mean = torch.load('model_for_realtime/input_mean.pt')
input_std = torch.load('model_for_realtime/input_std.pt')
target_mean = torch.load('model_for_realtime/target_mean.pt')
target_std = torch.load('model_for_realtime/target_std.pt')


def input_normalization(input_tensor):
    normalized_input = (input_tensor - input_mean) / input_std
    return normalized_input


def output_denormalization(np_array):
    denormalized_output = np_array * target_std.numpy() + target_mean.numpy()

    output_clipped = np.clip(denormalized_output, a_min=0, a_max=None)
    return output_clipped


def imageOut(filename, _input, _output, max_val=40, min_val=0):
    output = np.copy(_output)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    last_channel = _input[-1, -1, :, :]
    last_channel = last_channel * input_std[-1] + input_mean[-1]
    last_channel = np.delete(last_channel, [4 * i + 3 for i in range(16)], axis=1)
    last_channel = np.concatenate((last_channel, np.zeros((32, 16))), axis=1)
    last_channel_image = np.reshape(last_channel, (32, 64))
    ax1.set_aspect('equal', 'box')
    im1 = ax1.imshow(last_channel_image, cmap='jet', vmin=0, vmax=2500)
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1)

    ax2.set_aspect('equal', 'box')
    output_image = np.reshape(output, (32, 64))
    im2 = ax2.imshow(output_image, cmap='jet', vmin=min_val, vmax=max_val)
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    save_path = os.path.join(filename)
    plt.savefig(save_path)
    plt.close(fig)

def load_model(): #20240112 CJ
    # load trained CRNN model
    output_dir = "./TEST"
    os.makedirs(output_dir, exist_ok=True)
    
    netG = CRNN(channelExponent=4, dropout=0.0)
    doLoad = "model_for_realtime/CRNN_expo4_mean_01_10000model"
    if len(doLoad) > 0:
        netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    netG.to(device)
    netG.eval()
    return netG

# test model loaded
# layer1 = netG.layer1
# layer1_conv = layer1.layer1_conv
# print("Parameters of layer 'layer1_conv':")
# for param in layer1_conv.parameters():
#     print(param.size())
#     print(param.data)


# Artificially given some rows for testing
# pressure_lines = [
#     "[13:37:28.297]AA23FF0F4F0DFF0FB70F110FFF0FFF0FA30EFF0FDD093B08AF05380E140C1B08680E55",
#     "[13:37:28.390]AA23FF0F470DFF0FD30F0C0FFF0FFF0F9F0EFF0FDF093B08B305330E1D0C1A08690E55",
#     "[13:37:28.484]AA23FF0F4B0DFF0FD10F130FFF0FFF0FA60EFF0FDB093A08AD05450E1F0C18086F0E55",
#     "[13:37:28.593]AA23FF0F570DFF0FC60F0F0FFF0FFF0FA30EFF0FE3093E08B505350E110C0F081C0E55",
#     "[13:37:28.687]AA23FF0F530DFF0FB50F0F0FFF0FFF0FA30EFF0FE7094908AF052C0E050C0708680E55",
#     "[13:37:28.780]AA23FF0F4F0DFF0FDD0F130FFF0FFF0F9F0EFF0FE5093E08B5053C0E1F0C0508440E55",
#     "[13:37:28.888]AA23FF0F550DFF0FDB0F120FFF0FFF0F9C0EFF0FE3093708B4052D0E1F0CFF07170E55",
#     "[13:37:28.982]AA23FF0F4A0DFF0FCF0F0D0FFF0FFF0F9C0EFF0FE5093D08B9052F0E2E0C07082D0E55",
#     "[13:37:29.092]AA23FF0F3B0DFF0FD30F090FFF0FFF0F9C0EFF0FDF093708B405380E1C0C1208350E55",
#     "[13:37:29.185]AA23FF0F3F0DFF0FD60F0B0FFF0FFF0FA30EFF0FE5092D08B5053F0E0F0C1C08340E55",
#     "[13:37:29.279]AA23FF0F3D0DFF0FC70F0D0FFF0FFF0FA70EFF0FDD092B08B305370EFD0B1F083D0E55",
#     "[13:37:29.388]AA23FF0F3C0DFF0FBD0F0B0FFF0FFF0F9B0EFF0FE3092E08AE052F0E110C0B08330E55",
#     "[13:37:29.482]AA23FF0F2D0DFF0FBF0F0D0FFF0FFF0FA40EFF0FD7092F08AE052F0E0B0C0B08370E55"
# ]

# sleep_lines = [
#     "[13:37:28.530]AB11000000000000FC1300000097093455",
#     "[13:37:29.545]AB110000000000000E1400000099093455",
#     "[13:37:30.538]AB11000000000000E31300000097093455",
#     "[13:37:31.539]AB110000000000002C1400000097093955"
# ]

# List to store individual tensors, maintaining a rolling window of ten tensors
global_tensor_list = []
i = 0


def result_of_CRNN(pressure_line, sleep_line):
    global global_tensor_list

    pressure_time, pressure_value = process_pressure_values(pressure_line)
    sleep_time, sleep_value = process_sleep_values(sleep_line)

    pressure_seconds = pressure_time.split(':')[2].split('.')[0]
    sleep_seconds = sleep_time.split(':')[2].split('.')[0]

    if pressure_seconds == sleep_seconds:
        input_tensor = change_input_dimension(pressure_value, sleep_value)
        normalized_input = input_normalization(input_tensor)

        # If the list already has ten tensors, remove the first one
        if len(global_tensor_list) >= 10:
            global_tensor_list.pop(0)

        # Add the new tensor to the list
        global_tensor_list.append(normalized_input)

        if len(global_tensor_list) == 10:
            combined_tensor = torch.stack(global_tensor_list, dim=0)  # [10, 12, 32, 64]
            # print(combined_tensor[:, :, 1, 1])
            combined_tensor = combined_tensor.clone().unsqueeze(0)  # [1, 10, 12, 32, 64]
            # print(f"Combined tensor shape: {combined_tensor.shape}")

            output = netG(combined_tensor)
            output[output < 0] = 0
            # print(f"Output tensor shape: {output.shape}")
            denormalized_output = output_denormalization(output.detach().numpy())

            return combined_tensor, denormalized_output

    return None

def LowPressureData2img(model,pressure_lines,sleep_lines):  #20240112 CJ
    result = result_of_CRNN(model,pressure_lines, sleep_lines)
    if result is not None:
        input_tensor, output_denormalize = result
        output_image = np.reshape(output_denormalize[0], (32, 64))
        output_image1=cv2.resize(output_image,(640,320))
        cv2.imshow('pressure distribution', output_image1)
        #cv2.imshow(output_image)

        os.chdir("./TEST/")
        current_time = datetime.datetime.now()
        strdispaly=current_time.strftime("%Y-%m-%d_%H-%M-%S")
        imageOut(strdispaly, input_tensor[0], output_denormalize[0])
        os.chdir("../")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return      

# for pressure_line in pressure_lines:    #20240112 CJ
#     for sleep_line in sleep_lines:
#         result = result_of_CRNN(pressure_line, sleep_line)

#         if result is not None:
#             input, output = result

#             os.chdir("./TEST/")
#             imageOut("%04d" % i, input[0], output[0])
#             os.chdir("../")
#             i += 1

#             print(i)
