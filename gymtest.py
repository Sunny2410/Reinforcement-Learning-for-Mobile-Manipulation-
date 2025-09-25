import mujoco
import mujoco.viewer

# Đường dẫn đến file XML của bạn
xml_path = "/home/sunny24/rl_mm/asset/robot.xml"

# Load model và tạo dữ liệu
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Mở viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Đang hiển thị mô hình. Nhấn ESC để thoát.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
