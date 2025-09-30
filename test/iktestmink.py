import numpy as np
import mujoco
import mink

# 1. Load model
mj_model = mujoco.MjModel.from_xml_path("/home/sunny24/rl_mm/asset/SO101/so101_new_calib.xml")

# 2. Create Mink configuration
configuration = mink.Configuration(mj_model)

# 3. Optional: initialize from a keyframe (if available)
# configuration.update_from_keyframe("home")  # if your model has a "home" keyframe

# If no keyframe, Mink will start at zeros by default:
configuration.update(np.zeros(configuration.model.nq))  # sets joint positions

# 4. Define IK task
task = mink.FrameTask(
    "gripperframe",
    "site",
    position_cost=1.0,
    orientation_cost=1.0,
)
# Lấy transform hiện tại từ gripper
transform_init_to_world = configuration.get_transform_frame_to_world("gripperframe", "site")

# Tạo rotation bằng RPY (roll=0, pitch=0, yaw=90° quanh z)
rotation_so3 = mink.SO3.from_rpy_radians(roll=0, pitch=0, yaw=0)

# Tạo translation vector
translation_vec = np.array([-0.05, 0.0, 0])

# Tạo target transform SE3 với rotation + translation
transform_target_to_world = transform_init_to_world @ mink.SE3.from_rotation_and_translation(rotation_so3, translation_vec)

# Gán target cho IK task
task.set_target(transform_target_to_world)

# 5️⃣ Differential IK loop
dt = 0.01
velocity_tol = 1e-4
max_steps = 100
success = False

for step in range(max_steps):
    # Giải IK
    v = mink.solve_ik(configuration, [task], dt=dt, solver="daqp")
    
    if v is None or np.any(np.isnan(v)):
        print(f"Step {step}: IK solver returned invalid velocity, stopping loop.")
        break

    # Cập nhật Mink configuration
    configuration.integrate_inplace(v, dt)

    # Kiểm tra lỗi task
    err_vec = task.compute_error(configuration)  # 6D vector: [dx, dy, dz, dRx, dRy, dRz]

    pos_err = np.linalg.norm(err_vec[:3])  # position error magnitude
    rot_err = np.linalg.norm(err_vec[3:])  # rotation error magnitude

    print(f"Position error: {pos_err:.6f}, Rotation error: {rot_err:.6f}")

    print(f"Step {step}, error: {err_vec}")

    if np.linalg.norm(err_vec) < 1e-3:
        success = True
        break

if success:
    print("IK solved successfully.", configuration.q)
else:
    print("IK did not converge within max_steps or solver failed, but continuing anyway.")
