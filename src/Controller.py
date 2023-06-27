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
        self.action("LEFT BARKE",-1)
        self.action("RIGHT BARKE",-1)
        self.action("THROTTLE",-1)
        self.action("RUDDER",0)
    
    def initSim(self):
        self.action("CAMERA",0)
        time.sleep(0.1)
        self.action("INFOBAR",0)
        self.action("INFOBAR",0)

    def action(self,action,value):
        match action:
            case "RUDDER":
                self.gp.left_joystick_float(x_value_float=value, y_value_float=self.memljY)
                self.memljX = value
                self.gp.update()
                time.sleep(0.01)

            case "THROTTLE":
                self.gp.left_joystick_float(x_value_float=self.memljX, y_value_float=value)
                self.memljY = value
                self.gp.update()
                time.sleep(0.01)

            case "LEFT BARKE":
                self.gp.right_joystick_float(x_value_float=self.memrjX, y_value_float=value)
                self.memrjY = value
                self.gp.update()
                time.sleep(0.01)

            case "RIGHT BRAKE":
                self.gp.right_joystick_float(x_value_float=value, y_value_float=self.memrjY)
                self.memrjX = value
                self.gp.update()
                time.sleep(0.01)

            case "RESTART":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
                self.gp.update()
                time.sleep(0.2)

            case "ESC":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
                self.gp.update()
                time.sleep(0.2)

            case "SIM SPEEED ACC":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                self.gp.update()
                time.sleep(0.2)

            case "SIM SPEEED DCC":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                self.gp.update()
                time.sleep(0.2)

            case "SIM SPEEED RESET":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                self.gp.update()
                time.sleep(0.2)
            
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

            case "INFOBAR":
                self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
                self.gp.update()
                time.sleep(0.2)
                self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
                self.gp.update()
                time.sleep(0.2)  

        self.gp.update()