import argparse

parser = argparse.ArgumentParser(description="TP2 image generator.")

parser.add_argument("-i", "--image", type=str, help="Image target")
parser.add_argument("-t", "--triangles", type=int, help="Number of triangles")

def main() -> None:
    args = parser.parse_args()

    if args.image and args.triangles:
        print(f"Image {args.image} with {args.triangles} triangles")
    else:
        print("Missing args")
