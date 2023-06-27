import vgamepad as vg
import time

gamepad = vg.VX360Gamepad()

t = 0
rudder = 0
throttle = 0
dr = 100
dt = 10
gamepad.left_joystick(x_value=-32768, y_value=0)
gamepad.update()

while True:
    t += 1
    time.sleep(0.01)

    rudder = rudder + dr
    if rudder > 32768:
        rudder = 32700
        dr = -dr
    elif rudder < -32768:
        rudder = -32700
        dr = -dr

    throttle = throttle + dt
    if throttle > -15000:
        throttle = -15000
        dt = -dt
    elif throttle < -30000:
        throttle = -30000
        dt = -dt

    print(rudder,throttle)
    gamepad.left_joystick(x_value=rudder, y_value=throttle)
    gamepad.update()

