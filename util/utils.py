import numpy as np
import pickle, struct, socket, math
import zlib


def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return [i]


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    compressed = zlib.compress(msg_pickle, level=3)  # 折中压缩级别，避免CPU过高[2,7](@ref)

    # 优化2：分块发送避免大包传输
    total_size = len(compressed)
    sock.sendall(struct.pack(">I", total_size))  # 先发送总长度

    # 分块发送压缩数据（每块1MB）
    chunk_size = 1024 * 1024
    for i in range(0, total_size, chunk_size):
        chunk = compressed[i:i + chunk_size]
        sock.sendall(chunk)  # 避免一次性发送大包[4](@ref)

    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None, timeout=None):
    # 设置超时
    if timeout is not None:
        sock.settimeout(timeout)
    
    try:
        msg_len = struct.unpack(">I", sock.recv(4))[0]

        # 优化3：流式接收避免内存爆炸
        received_chunks = []
        bytes_received = 0
        while bytes_received < msg_len:
            chunk = sock.recv(min(4096, msg_len - bytes_received))
            if not chunk:
                raise RuntimeError("连接已关闭，无法接收完整消息")
            received_chunks.append(chunk)
            bytes_received += len(chunk)

        compressed_msg = b''.join(received_chunks)

        # 优化4：添加解压缩异常处理
        try:
            msg_pickle = zlib.decompress(compressed_msg)
        except zlib.error as e:
            raise RuntimeError(f"解压失败: {e}")  # 捕获zlib错误[2](@ref)

        msg = pickle.loads(msg_pickle)
        print(msg[0], 'received from', sock.getpeername())

        if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
            raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg
    except socket.timeout:
        raise RuntimeError("接收消息超时")
    except struct.error:
        raise RuntimeError("消息格式错误，无法解析")
    except Exception as e:
        raise RuntimeError(f"接收消息时发生错误: {str(e)}")
    finally:
        # 恢复默认超时设置
        if timeout is not None:
            sock.settimeout(None)


def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr


def get_indices_each_node_case(n_nodes, maxCase, label_list):
    indices_each_node_case = []

    for i in range(0, maxCase):
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1

    for i in range(0, len(label_list)):
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i)

        # case 2
        tmp_target_node = int((label_list[i] - min_label) % n_nodes)
        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        for n in range(0, n_nodes):
            indices_each_node_case[2][n].append(i)

        # case 4
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2))
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp
        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[3][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[3][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[3][tmp_target_node].append(i)

    return indices_each_node_case
