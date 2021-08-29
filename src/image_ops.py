BATCH_SIZE = 64
TARGET_SIZE = (300, 300)

def preprocess_data(images, labels):
    images = (images - 127.00) / 128.00  # [0;255] -> [0;1]
    return images, labels