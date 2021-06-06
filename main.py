from filter import ImageHolder
import multiprocessing
import os

IMAGE_DIR = "images"


if __name__ == "__main__":
    print("Virtual cores:", multiprocessing.cpu_count())
    ImH = ImageHolder()
    processes = []
    for i, filename in enumerate(os.listdir(IMAGE_DIR)):
        processes += [multiprocessing.Process(target=ImH.apply_filters, args=(IMAGE_DIR + '/' + filename,))]

    processes_run = []
    stack = []
    for proc in processes:
        proc.start()
