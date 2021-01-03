import csv
import json
import os


def read_data(input_path):
    return_data = []

    ext = os.path.splitext(input_path)[-1].lower()

    if ext != ".csv":
        raise Exception("The input file is not a csv file")

    print("Reading the input file")
    with open(input_path, "r") as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            return_data.append(row[0])

    return return_data


def write_data(output_path, sentence, data):
    # ext = os.path.splitext(output_path)[-1].lower()
    # if ext != ".csv":
    #     raise Exception("The input file is not a csv file")

    with open(output_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([sentence,data])


def get_params(config_path):

    ext = os.path.splitext(config_path)[-1].lower()
    if ext != ".json":
        raise Exception("The config file is not a json file")

    print("Parsing your config file")
    with open(config_path, "r") as jsonfile:
        data = json.load(jsonfile)
        return data