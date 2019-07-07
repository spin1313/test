import numpy as np
import keras
from PIL import Image
import socket
import sys


class SocketCommunication:

    def __init__(self, host="localhost", port=50000):
        self.serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverSock.bind((host, port))  # IPとPORTを指定してバインドします
        self.serverSock.listen(10)  # 接続の待ち受けをします（キューの最大数を指定）
        print('Waiting for connections...')
        self.clientSock, self.client_address = self.serverSock.accept()  # 接続されればデータを格納
        print('Connection accept.')

    def receive(self):
        while True:
            rcv_msg = self.clientSock.recv(4096)
            rcv_msg = rcv_msg.decode('utf-8')
            if rcv_msg != '':
                print('Received -> %s' % rcv_msg)
                return rcv_msg

    def send(self, msg):
        msg = msg.encode('utf-8')
        if msg == '':
            print('No msg')
            return
        self.clientSock.sendall(msg)  # メッセージを返します
        print('Send -> %s' % msg)

    def close(self):
        self.clientSock.close()


class InferenceServer:
    def __init__(self):
        self.com = SocketCommunication()
        self.model = None
        self.model_path = None

    def run_command_manager(self):
        print('Waiting command...')
        # コマンド受信
        msg = self.com.receive()
        print('Command Received...')
        msg = str(msg).rstrip()  # 末尾の空白文字削除

        # 結果送信
        send_msg = self.command_execution(msg)
        print('Command Done...')
        self.com.send(send_msg)
        print('Result Send...')

    def command_execution(self, msg):
        try:
            command_str = msg.split(",")
            if command_str[0] == "LOAD_MODEL":
                self.model_path = command_str[1]
                print(self.model_path)
                self.load_model(self.model_path)
                result = "SUCCESS"
            elif command_str[0] == "INFERENCE":
                if command_str[1] == "IMAGE_TO_CLASS":
                    path_list = command_str[2:]
                    result = self.inference_image_to_class(path_list)
                else:
                    raise Exception("Invalid command format")
            else:
                raise Exception("Invalid command format")
        except Exception as e:
            print(e)
            result = "ERROR"
        return result

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        print("Model load success...")

    def inference_image_to_class(self, path_list):
        img = Image.open(path_list[0])  # 先頭ファイルを読み込み
        img = np.asarray(img)
        img_array = np.ndarray([len(path_list), *img.shape])  # 画像データ格納用ndarray
        for i, file in enumerate(path_list):
            img_array[i] = np.asarray(Image.open(file))
        print("Image load success...")

        # img_arrayをTensorFlow向けにreshape
        if img_array.ndim == 3:
            # モノクロ画像の場合はCh次元を追加
            img_array = img_array.reshape(*img_array.shape, 1)
        img_array = img_array/255
        print("Image shape:" + str(img_array.shape))

        # 推論
        inf_result = self.model.predict(img_array)
        print(inf_result)
        class_num = inf_result.shape[1]  # クラス数を取得

        # 推論結果文字列生成
        result_str = ""
        for element in inf_result.flatten():
            result_str += "{:.4f}".format(element) + ","
        result_str = "SUCCESS," + str(class_num) + "," + result_str[:-1]  # 末尾のカンマを削除
        return result_str


if __name__ == "__main__":
    args = sys.argv

    print(args)
    print("第1引数：" + args[1])
    print("第2引数：" + args[2])
    print("第3引数：" + args[3])
    server = InferenceServer()
    while True:
        server.run_command_manager()
