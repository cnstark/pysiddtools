import os
import requests
import threading

CURRENT_DIR = os.path.dirname(__file__)
URLS_FILE_PATH = os.path.join(CURRENT_DIR, "SIDD_URLs_Mirror_2.txt")


def format_dataset(base_dir):
    data_dir_list = os.listdir(base_dir)
    for data_dir in data_dir_list:
        data_dir_full = os.path.join(base_dir, data_dir)
        file_list = os.listdir(data_dir_full)
        if len(file_list) == 1:
            print(data_dir)
            print(file_list)
            cmd = "mv " + os.path.join(data_dir_full, file_list[0], "*") + " " + data_dir_full
            os.system(cmd)
            cmd = "rm -r " + os.path.join(data_dir_full, file_list[0])
            os.system(cmd)


def wget(file_url, file_name, save_dir=None):
    if save_dir is not None:
        file_name = os.path.join(save_dir, file_name)
    os.system("wget -t0 -c -O " + file_name + " " + file_url)


def download_sidd(save_dir=None):
    def download(url_list, save_dir=None):
        for index, url in enumerate(url_list):
            h = requests.head(url)
            file_url = h.headers['Location']
            file_name = file_url.split('/')[-1].split('?')[0]
            print('thread: ', threading.current_thread().name, ' downloading ', str(index), ' file: ', file_name)
            wget(url, file_name, save_dir)

    with open(URLS_FILE_PATH, 'r') as f:
        urls = f.readlines()
        print(len(urls))
        for index, i in enumerate(range(0, len(urls), 10)):
            urls_thread = urls[i: i + 10]
            t = threading.Thread(target=download, args=(urls_thread, save_dir), name=str(index))
            t.start()
