from filter import ImageHolder
import multiprocessing
import os


IMAGE_DIR = "images"


# stack = []
# stack_full = True
#
# while stack_full:


if __name__ == "__main__":
    print("Virtual cores:", multiprocessing.cpu_count())
    ImH = ImageHolder()
    processes = []
    for i, filename in enumerate(os.listdir(IMAGE_DIR)):
        processes += [multiprocessing.Process(target=ImH.apply_filters, args=(IMAGE_DIR+'/'+filename,))]

    processes_run = []
    stack = []
    for proc in processes:
        proc.start()


    #
    # processes_finished = []
    # while processes:
    #     for i, proc in enumerate(processes):
    #         if i >= multiprocessing.cpu_count():
    #             break
    #         if not proc.is_alive():
    #             processes_finished += [proc]
    #             del stack[i]
    #             del processes[i]
    #             stack.append()
    #             break
    #
    #
    # # print(123)
    # if processes[0].is_alive():
    #     print(1)
    # while processes[0].is_alive():
    #     pass
    # print(123)
