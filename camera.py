
import cv2
import depthai as dai
from pathlib import Path

IMAGE_PATH = Path("images/bookshelf.jpg")


def capture_bookshelf():
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with dai.Pipeline() as pipeline:

        # Use the OAK-D Lite RGB camera
        camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )

        # Request a high-resolution image
        queue = camera.requestOutput(
            (1920, 1080), fps=15
        ).createOutputQueue()

        pipeline.start()

        print("OAK-D Lite camera started")
        print("Press S to save an image")
        print("Press Q to quit")

        while pipeline.isRunning():
            frame = queue.get().getCvFrame()

            cv2.imshow("Smart Bookshelf Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                cv2.imwrite(str(IMAGE_PATH), frame)
                print(f"Image saved: {IMAGE_PATH}")
                return str(IMAGE_PATH)

            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    return None
