import re

def parse_multirotor_state(state_str):
    # Define regular expressions for each attribute
    patterns = {
        "position": '(?<=position = \{)(.*?)(?=\})',
        "orientation": '(?<=orientation = \{)(.*?)(?=\})',
        "linear_velocity": '(?<=linear_velocity = \{)(.*?)(?=\})',
        "angular_velocity": '(?<=angular_velocity = \{)(.*?)(?=\})',
        "rpm": '(?<=rpm = \{)(.*?)(?=\})',
        "force": '(?<=force = \{)(.*?)(?=\})',
        "torque": '(?<=torque = \{)(.*?)(?=\})'
    }

    # Initialize dictionary to store extracted data
    extracted_data = {}

    # Iterate over patterns and extract data
    for key, pattern in patterns.items():
        match = re.search(pattern, state_str)
        if match:
            # Split the matched string by comma and convert to float
            values = match.group(1).split(", ")
            extracted_data[key] = [float(value) for value in values]

    return extracted_data
import numpy as np
def dict_to_array(data):
    # Extract and flatten the values for the first tensor
    tensor1_values = (
        data['position'] +
        data['orientation'] +
        data['linear_velocity'] +
        data['angular_velocity'] +
        data['rpm']
    )
    tensor1 = np.array(tensor1_values)

    # Extract and flatten the values for the second tensor
    tensor2_values = data['force'] + data['torque']
    tensor2 = np.array(tensor2_values)
    return tensor1, tensor2

def parse_log_file(file_path):
    # Read the log file
    states = []
    next_states = []
    actions = []
    rewards = []
    with open(file_path, 'r') as file:
        # log_data = file.read()
        end = False
        while not end:
            line = file.readline()
            if "next_state" in line:
                next_state_dict = parse_multirotor_state(line.strip('next_state = '))

                next_states.append(next_state_dict)
            elif "state" in line:
                state_dict = parse_multirotor_state(line.strip('state = '))
                states.append(state_dict)
            elif "action" in line:
                action = line.strip('action = ')
                actions.append([float(x) for x in action.strip('{}\n').split(', ')])
            elif "reward" in line:
                reward = line.strip('reward = ')
                rewards.append(float(reward))
            if 'file_end' in line:
                end = True

    print(len(actions))
    xs = np.zeros((17,9))
    xs_next = np.zeros((17,9))
    disturbances = np.zeros((6,9))
    a = np.zeros((4,9))
    r = np.zeros((1,9))
    for i in range(9):
        state = states[i]
        next_state = next_states[i]
        action = actions[i]
        reward = rewards[i]
        xs[:,i], disturbances[:,i] = dict_to_array(state)
        xs_next[:,i], _ = dict_to_array(next_state)
        a[:,i] = action
        r[0,i] = reward

    return xs, xs_next, disturbances, a, r, [states,next_states, actions, rewards]
    #     extracted_data_lst.append(extracted_data)

    # return extracted_data_lst
if __name__ == "__main__":
    state_str_lst = ["""{<rl_tools::rl::environments::multirotor::StateRotors<float, unsigned long, rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> > >> = {<rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> >> = {<rl_tools::rl::environments::multirotor::StateBase<float, unsigned long>> = {        static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor9StateBaseIfmE20REQUIRES_INTEGRATIONE".>, static DIM = 13, position = {0.106121972, -0.00812661648, -0.184007213}, orientation = {0.780337036, -0.408706337, 0.427678913, -0.202790499},         linear_velocity = {-0.769604683, -0.745716214, 0.747151971}, angular_velocity = {-0.616498947, 0.501791358, -0.392133534}},       static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor16StateRandomForceIfmNS2_9StateBaseIfmEEE20REQUIRES_INTEGRATIONE".>, static DIM = 19, force = {-0.00529993186, 0.0193259362, -0.00569333183}, torque = {-1.27824194e-06, -2.44328221e-05,         2.11097067e-05}}, static REQUIRES_INTEGRATION = true, static PARENT_DIM = 19, static DIM = 23, rpm = {10851, 10851, 10851, 10851}},   static REQUIRES_INTEGRATION = true, static HISTORY_LENGTH = 32, static PARENT_DIM = 19, static action_DIM = <optimized out>, static DIM = 23,  action_history = {{0, 0, 0, 0} <repeats 32 times>}}""",
                    """{<rl_tools::rl::environments::multirotor::StateRotors<float, unsigned long, rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> > >> = {<rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> >> = {<rl_tools::rl::environments::multirotor::StateBase<float, unsigned long>> = {        static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor9StateBaseIfmE20REQUIRES_INTEGRATIONE".>, static DIM = 13, position = {0.0986464322, -0.015419716, -0.176954553}, orientation = {0.77767694, -0.411438018, 0.42936942, -0.203909919},         linear_velocity = {-0.725437045, -0.712875068, 0.663328111}, angular_velocity = {-0.61561507, 0.443669081, -0.367664814}},       static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor16StateRandomForceIfmNS2_9StateBaseIfmEEE20REQUIRES_INTEGRATIONE".>, static DIM = 19, force = {-0.00529993186, 0.0193259362, -0.00569333183}, torque = {-1.27824194e-06, -2.44328221e-05,         2.11097067e-05}}, static REQUIRES_INTEGRATION = true, static PARENT_DIM = 19, static DIM = 23, rpm = {10953.4131, 10804.3604, 10967.751, 10802.6982}},   static REQUIRES_INTEGRATION = true, static HISTORY_LENGTH = 32, static PARENT_DIM = 19, static action_DIM = <optimized out>, static DIM = 23,   action_history = {{0, 0, 0, 0} <repeats 31 times>, {0.146342993, -0.0666453615, 0.16683118, -0.0690206438}}}""",
                    """{<rl_tools::rl::environments::multirotor::StateRotors<float, unsigned long, rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> > >> = {<rl_tools::rl::environments::multirotor::StateRandomForce<float, unsigned long, rl_tools::rl::environments::multirotor::StateBase<float, unsigned long> >> = {<rl_tools::rl::environments::multirotor::StateBase<float, unsigned long>> = {        static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor9StateBaseIfmE20REQUIRES_INTEGRATIONE".>, static DIM = 13, position = {0.0916144997, -0.0223835297, -0.170741692}, orientation = {0.775140464, -0.414172918, 0.430867463, -0.204869017},         linear_velocity = {-0.68088758, -0.679857731, 0.57919389}, angular_velocity = {-0.60920769, 0.394342661, -0.36418739}},       static REQUIRES_INTEGRATION = <error reading variable: Missing ELF symbol "_ZN8rl_tools2rl12environments10multirotor16StateRandomForceIfmNS2_9StateBaseIfmEEE20REQUIRES_INTEGRATIONE".>, static DIM = 19, force = {-0.00529993186, 0.0193259362, -0.00569333183}, torque = {-1.27824194e-06, -2.44328221e-05,         2.11097067e-05}}, static REQUIRES_INTEGRATION = true, static PARENT_DIM = 19, static DIM = 23, rpm = {11048.8545, 10761.6924, 11077.4004, 10755.2959}},   static REQUIRES_INTEGRATION = true, static HISTORY_LENGTH = 32, static PARENT_DIM = 19, static action_DIM = <optimized out>, static DIM = 23,   action_history = {{0, 0, 0, 0} <repeats 30 times>, {0.146342993, -0.0666453615, 0.16683118, -0.0690206438}, {0.145819575, -0.065268144, 0.16744265,       -0.0721870512}}}""",
    ]

    # for state_str in state_str_lst:
    #     print(parse_multirotor_state(state_str))

    log_file_path = "/home/korneel/learning_to_fly/learning_to_fly/learning-to-fly/include/learning_to_fly/simulator/log.txt"
    extracted_data_lst = parse_log_file(log_file_path)
    print(extracted_data_lst[0])
    print(extracted_data_lst[1])
    print(extracted_data_lst[2])
    print(extracted_data_lst[3])
    print(extracted_data_lst[4])