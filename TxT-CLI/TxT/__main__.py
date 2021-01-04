import sys
import os
import csv

from .model import GlobalModel
from .file_parser import write_data, read_data


def main():
    args = sys.argv[1:]

    # No of arguments check
    if len(args) != 3:
        raise Exception("The input format should be <option> <inputfilename> <outputfilename>")


    option = args[0]
    inputPath = args[1]
    outputPath = args[2]

    data = read_data(inputPath)

    model = GlobalModel()

    if(option == "tsa"):

        print("Running the model ...")

        for row in data:
            output = model.tsaForward(row)
            write_data(outputPath, row, output)

        print("Finished writing ...")

    elif(option == "sum"):
        pass


    elif(option == "para"):
        pass


if __name__ == '__main__':
    main()