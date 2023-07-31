import os

relative_path = "../../Model/Guineapig/resnetxSVM/resnet_EXTRACTOR.h5"
absolute_path = os.path.abspath(relative_path)

if os.path.exists(absolute_path):
    print("The file exists at:", absolute_path)
else:
    print("The file does not exist at:", absolute_path)