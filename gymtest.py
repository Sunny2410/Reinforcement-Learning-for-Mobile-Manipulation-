import mujoco
import mujoco.viewer
import numpy as np

# Đường dẫn đến file XML của robot
xml_path = "/home/sunny24/rl_mm/asset/robotfull.xml"

# Load model và tạo dữ liệu
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Lấy id của end-effector (site gripperframe)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
print("ee_id:", ee_id)

def get_fk(model, data, site_id):
    """Trả về (pos, mat) của 1 site từ FK trong world frame"""
    mujoco.mj_forward(model, data)  # update kinematics
    pos = np.copy(data.site_xpos[site_id])
    mat = np.copy(data.site_xmat[site_id].reshape(3, 3))
    return pos, mat

# Mở viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Đang hiển thị mô hình. Nhấn ESC để thoát.")
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Lấy FK của end-effector (trong world frame)
        pos, mat = get_fk(model, data, ee_id)
        print("EE pos (world):", pos)
        # print("EE mat (world):\n", mat)

        viewer.sync()
