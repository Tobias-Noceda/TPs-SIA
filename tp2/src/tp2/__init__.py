import argparse
import json

parser = argparse.ArgumentParser(description="Genetic image generator.")

parser.add_argument("-i", "--image", type=str, required=True, help="Image target")
parser.add_argument("-t", "--triangles", dest="n", type=int, required=True, help="Number of triangles")
parser.add_argument("-p", "--hyper", dest="params", type=str, required=True, help="Hyperparameters JSON file")

def main() -> None:
    args = parser.parse_args()

    if args.image and args.n and args.params:
        with open(args.params, 'r') as file:
            data = json.load(file)
        print(f"Image {args.image} with {args.n} triangles and {data} params")
    else:
        print("Missing args")
