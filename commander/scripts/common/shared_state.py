class SharedState:
    def __init__(self):
        self.paused = False
        self.cart_pose_x = 0
        self.pole_pitch = 0
        self.pole_tip_pose_z = 0