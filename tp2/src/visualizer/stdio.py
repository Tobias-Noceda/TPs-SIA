import json
from multiprocessing import shared_memory

SHM_NAME = "visualizer_state"
SHM_SIZE = 4096

def open_shm():
    try:
        shm = shared_memory.SharedMemory(
            name=SHM_NAME,
            create=True,
            size=SHM_SIZE,
        )
    except FileExistsError:
        old_shm = shared_memory.SharedMemory(name=SHM_NAME)
        old_shm.unlink()
        old_shm.close()

        shm = shared_memory.SharedMemory(
            name=SHM_NAME,
            create=True,
            size=SHM_SIZE,
        )

    return shm

def write_state(shm: shared_memory.SharedMemory, state: dict):
    data = json.dumps(state).encode("utf-8")

    if len(data) > SHM_SIZE:
        raise ValueError("State is too large for shared memory")

    if shm.buf is None:
        raise BufferError("Shared memory buffer is closed")

    shm.buf[:4] = len(data).to_bytes(4, "little")
    shm.buf[4:4 + len(data)] = data

def main():
    shm = open_shm()
    print(f"Shared memory created: {SHM_NAME}")

    try:
        counter = 0
        triangles = []

        while True:
            a = list(map(int, input("Ax,Ay: ").split(",")))
            b = list(map(int, input("Bx,By: ").split(",")))
            c = list(map(int, input("Cx,Cy: ").split(",")))

            color = input("Color: ")

            triangles.append({
                "a": a,
                "b": b,
                "c": c,
                "color": color
            })

            state = {
                "counter": counter,
                "triangles": triangles,
            }

            write_state(shm, state)
            counter += 1

    finally:
        try:
            shm.close()
            shm.unlink()
        except:
            print("Ups")

if __name__ == "__main__":
    main()
