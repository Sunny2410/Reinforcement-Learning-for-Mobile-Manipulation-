from rl_mm.robots import MobileSO101
from dm_control import mjcf
import imageio
from rl_mm.arena import StandardArena
from rl_mm.props import Primitive
from rl_mm.utils.transform_utils import mat2quat
if __name__ == "__main__":
    # --- Tạo arena ---
    arena = StandardArena()
    
    # Ví dụ attach box
    box = Primitive(type="box", size=[0.02,0.02,0.02], pos=[0.5,0,0], rgba=[1,0,0,1])
    arena.attach_free(box.mjcf_model, pos=[0.5,0,0])

    # --- Load robot ---
    robot = MobileSO101(name="test_robot")
    print("Load robot successful:", robot)
    for joint in robot.joints_arm:
        print("Name:", joint.name)
        print("site:", robot.base_site.name)
    if getattr(robot, "baseframe", None) is not None: 
        print("Baseframe:")
    # Attach robot vào arena
    arena.attach(robot.mjcf_model, pos=[0,0,0])
    
    # --- Tạo physics từ arena ---
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    physics.forward()
    box_body_id = physics.model.name2id('unnamed_model/', 'body')

    # Lấy tọa độ của body trong không gian
    box_pos = physics.data.xpos[box_body_id]  # vector (x, y, z)
    box_quat = physics.data.xpos[box_body_id]  # quaternion (rotation)

    print("Attach box successful:", box)
    print("Box position:", box_pos)
    print("Box orientation (quaternion):", box_quat)

    # for i in range(physics.model.nbody):
    #     name = physics.model.id2name(i, 'body')
    #     print(f"Body {i}: {name}")

    # --- In ra tất cả site ---
    print(f"Number of sites: {physics.model.nsite}")
    for i in range(physics.model.nsite):
        name = physics.model.id2name(i, 'site')  # Lấy tên site theo index
        pos = physics.data.site_xpos[i].copy()   # Vị trí world
        quat = mat2quat(physics.data.site_xmat[i].copy().reshape(3, 3))

        print(f"Site {i}: {name}, position: {pos}, Pose: {quat}")
    
    # --- Render ảnh ---
    img = physics.render(height=480, width=640, camera_id=-1)
    imageio.imwrite("rl_mm/test/arena_robot.png", img)
    print("Saved image to arena_robot.png")
