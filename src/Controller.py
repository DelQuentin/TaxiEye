import vgamepad as vg
import time

class Controller:
    def __init__(self) -> None:
        self.gp = vg.VX360Gamepad()
        self.gp.reset()
        self.gp.update()
        self.memljX = 0
        self.memljY = 0
        self.memrjX = 0
        self.memrjY = 0

        # press a button to wake the device up
        self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gp.update()
        time.sleep(0.2)
        self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gp.update()
        time.sleep(0.2)

    def idle(self):
        self.action("LEFT BARKE",0)
        self.action("RIGHT BARKE",0)
        self.action("THROTTLE",0)
        self.action("RUDDER",0)
    
    def initSim(self):
        self.action("CAMERA",0)
        time.sleep(0.1)
        self.action("COORD",0)
        time.sleep(0.1)

    def action(self,action,value):
        match action:
            case "RUDDER":
                self.gp.left_joystick_float(x_value_float=value, y_value_float=self.memljY)
                self.memljX = value
                self.gp.update()
                time.sleep(0.01)

            case "THROTTLE":
                self.gp.left_joystick_float(x_value_float=self.memljX, y_value_float=-1+2*value)
                self.memljY = -1+2*value
                self.gp.update()
                time.sleep(0.01)

            case "LEFT BARKE":
                self.gp.right_joystick_float(x_value_float=self.memrjX, y_value_float=-1+2*value)
                self.memrjY = -1+2*value
                self.gp.update()
                time.sleep(0.01)

            case "RIGHT BRAKE":
                self.gp.right_joystick_float(x_value_float=-1+2*value, y_value_float=self.memrjY)
                self.memrjX = -1+2*value
                self.gp.update()
                time.sleep(0.01)
            
            case "PARK":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
                self.gp.update()
                time.sleep(0.2)
            
            case "NWS":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
                self.gp.update()
                time.sleep(0.2)

            case "CAMERA":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
                self.gp.update()
                time.sleep(0.2)

            case "COORD":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
                self.gp.update()
                time.sleep(0.2)

        self.gp.update()