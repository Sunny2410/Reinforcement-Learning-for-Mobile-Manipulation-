from rl_mm.props import Primitive
from rl_mm.robots import MobileSO101
from dm_control import mjcf
import imageio
import os


if __name__ == "__main__":
    # Tạo world root
    world = mjcf.RootElement()

    # Thêm robot
    robot = MobileSO101(name="test_robot")
    world.attach(robot.mjcf_model)

    # Thêm một khối hộp đỏ làm prop
    box = Primitive(type="box", size=[0.1, 0.1, 0.1], rgba=[1, 0, 0, 1])
    world.attach(box.mjcf_model)

    # Build physics từ world
    physics = mjcf.Physics.from_mjcf_model(world)

    # Render image (dùng free camera -1)
    img = physics.render(height=480, width=480, camera_id=-1)

    # Save ảnh
    save_path = "rl_mm/test/env.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, img)

    print("Environment with robot + primitive saved at:", save_path)
