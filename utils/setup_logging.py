
import logging
import sys
#* 同时输出到.log和控制台
def setup_logging(filename,file_level=logging.INFO,console_level=logging.INFO): # 注意缺省值INFO的大小写
    file_handler = logging.FileHandler(filename,mode = 'a')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt='%m/%d/%Y %H:%M:%S'))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt='%m/%d/%Y %H:%M:%S'))
    console_handler.setLevel(console_level)

    logging.basicConfig(level=min(file_level,console_level),handlers=[file_handler,console_handler])


if __name__ == '__main__':
    setup_logging(filename='./xxx.log')
    logging.info('hjw comes from UESTC')