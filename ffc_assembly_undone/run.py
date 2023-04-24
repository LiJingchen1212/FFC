from panda_env_ffc_assembly import PandaEnv_FFCAssembly

import pybullet as p
import numpy as np

A_ZERO = np.array([0,0,0])
SCALE_LIST = [0.01, 0.05, 0.1, 0.3] # speed
A_SCALE = SCALE_LIST[2]
A_X = np.array([1,0,0])
A_Y = np.array([0,1,0])
A_Z = np.array([0,0,1])

k_ZP = p.B3G_UP_ARROW
k_ZN = p.B3G_DOWN_ARROW
k_XP = p.B3G_RIGHT_ARROW
k_XN = p.B3G_LEFT_ARROW
k_YP = ord('x')
k_YN = ord('z')

k2a_dict = {k_ZP:A_Z, k_ZN:-A_Z, k_XP:A_X, k_XN:-A_X, k_YP:A_Y, k_YN:-A_Y}


def key2action(keys):
    action = A_ZERO
    for key in k2a_dict:
        if key in keys and keys[key] & p.KEY_IS_DOWN: # 按下会一直动
        # if key in keys and keys[key] & p.KEY_WAS_TRIGGERED: # 按下只会动一下
            action = k2a_dict[key]
            break
    return action

def key2other(keys):
    # press '1'/'2'/'3'/'4' to set speed
    global A_SCALE
    for idx, key in enumerate([ord('1'), ord('2'), ord('3'), ord('4')]):
        if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
            A_SCALE = SCALE_LIST[idx]
            print('action scale:', A_SCALE)
            break


panda = PandaEnv_FFCAssembly()


while True:
    keys = p.getKeyboardEvents()
    key2other(keys)
    action = A_SCALE * key2action(keys)

    if np.linalg.norm(action) > 0.001:
        panda.inc_ee_pos(action)

    pos = panda.get_ee_pos()
    force = panda.get_ee_force()
    
    print('pos = [{:.4f},{:.4f},{:.4f}], force = [{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.
          format(pos[0], pos[1], pos[2], 
                 force[0], force[1], force[2], 
                 force[3], force[4], force[5]))

    panda.get_image()
    panda.step()

