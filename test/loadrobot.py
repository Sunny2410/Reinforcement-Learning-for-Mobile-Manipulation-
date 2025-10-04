import mujoco
import mujoco.viewer

# Đường dẫn tới file XML
xml_path = "/home/sunny24/rl_mm/asset/SO101/so101_new_calibmink.xml"

# Tải mô hình
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Tạo viewer để render
viewer = mujoco.viewer.launch_passive(model, data)

# Vòng lặp mô phỏng
while True:
    mujoco.mj_step(model, data)  # bước mô phỏng
    viewer.sync()                # cập nhật render
