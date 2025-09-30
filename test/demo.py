import gymnasium
import rl_mm

env = gymnasium.make("rl_mm/SO101-v0", render_mode="human")
obs, info = env.reset(seed=42)

print("Nhập số action rồi nhấn Enter. Nhập q để thoát.")
idx = None

try:
    while True:
        # Nếu chưa có action đang chạy, đọc input mới
        if idx is None:
            key = input("Action index: ").strip()
            if key.lower() == 'q':
                break
            if key.isdigit():
                tmp_idx = int(key) - 1  # nếu muốn nhập 1..N thay vì 0..N-1
                if 0 <= tmp_idx < env.action_space.n:
                    idx = tmp_idx
                else:
                    print(f"Action {tmp_idx+1} không hợp lệ (1-{env.action_space.n})")
                    continue
            else:
                print("Nhập số từ 1 đến", env.action_space.n, "hoặc q để thoát")
                continue

        # Nếu có action, gọi step liên tục cho đến khi controller xong
        obs, reward, terminated, truncated, info = env.step(idx)
        env.render()
        
        if not env.manager.is_any_moving():  # controller xong
            idx = None

        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
