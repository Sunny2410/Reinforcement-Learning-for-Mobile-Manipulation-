from rl_mm.robots import MobileSO101
from dm_control import mjcf
import imageio

if __name__ == "__main__":
    robot = MobileSO101(name="test_robot")
    print("Load robot succesfull:", robot)
    print("Jointarm:", robot.joints_arm)
    print("Jointbase:", robot.joints_base)

    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    # Dùng free camera (-1) nếu model không có fixed camera
    img = physics.render(height=480, width=480, camera_id=-1)

    imageio.imwrite("rl_mm/test/robot.png", img)
    print("Image is saved")
