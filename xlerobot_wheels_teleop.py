#!/usr/bin/env python3
"""
Simple wheels-only teleoperation for XLerobot using keyboard.
No UI dependencies, just direct keyboard input for wheel control.

For real-time key detection (hold keys to move), install:
    pip install keyboard

Without the keyboard library, it falls back to Enter-based input.
"""

import sys
import time
import threading
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

KEYBOARD_AVAILABLE = False

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig


class SimpleKeyboardWheelsTeleop:
    """Simple keyboard teleoperation for XLerobot wheels only."""
    
    def __init__(self, robot_config):
        self.robot = XLerobot(robot_config)
        self.running = False
        self.current_action = {
            "x.vel": 0.0,
            "y.vel": 0.0, 
            "theta.vel": 0.0
        }
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium  
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0
        
        # Key states for real-time input
        self.pressed_keys = set()
        
    def connect(self):
        """Connect to robot."""
        print("Connecting to XLerobot...")
        self.robot.connect(calibrate=False)
        print("✅ Connected!")
        
    def disconnect(self):
        """Disconnect from robot."""
        print("Disconnecting...")
        self.robot.stop_base()
        self.robot.disconnect()
        print("✅ Disconnected!")
        
    def print_controls(self):
        """Print control instructions."""
        print("\n" + "="*50)
        print("🎮 XLEROBOT WHEELS TELEOPERATION")
        print("="*50)
        print("Controls:")
        print("  w/s  - Forward/Backward")
        print("  a/d  - Left/Right") 
        print("  q/e  - Rotate Left/Right")
        print("  n/m  - Speed Up/Down")
        print("  x    - Stop")
        print("  z    - Quit")
        print("="*50)
        print(f"Current speed: {self.speed_levels[self.speed_index]}")
        print("="*50)
        
    def update_speed(self, direction):
        """Update speed level."""
        if direction == "up":
            self.speed_index = min(self.speed_index + 1, 2)
        elif direction == "down":
            self.speed_index = max(self.speed_index - 1, 0)
        
        speed = self.speed_levels[self.speed_index]
        print(f"Speed: {speed['xy']:.1f} m/s, {speed['theta']}°/s")
        
    def update_action_from_keys(self):
        """Update action based on currently pressed keys."""
        speed = self.speed_levels[self.speed_index]
        
        # Reset action
        self.current_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        
        # Movement controls (WASD layout, 180° rotated)
        if 'w' in self.pressed_keys:
            self.current_action["x.vel"] = -speed["xy"]  # Forward is now backward
        if 's' in self.pressed_keys:
            self.current_action["x.vel"] = speed["xy"]   # Backward is now forward
        if 'a' in self.pressed_keys:
            self.current_action["y.vel"] = -speed["xy"]  # Left is now right
        if 'd' in self.pressed_keys:
            self.current_action["y.vel"] = speed["xy"]   # Right is now left
        if 'q' in self.pressed_keys:
            self.current_action["theta.vel"] = -speed["theta"]  # Rotate left
        if 'e' in self.pressed_keys:
            self.current_action["theta.vel"] = speed["theta"]   # Rotate right
            
    def on_key_press(self, key):
        """Handle key press events."""
        try:
            key_name = key.name.lower()
            if key_name in ['w', 'a', 's', 'd', 'q', 'e']:
                self.pressed_keys.add(key_name)
            elif key_name == 'n':
                self.update_speed("up")
            elif key_name == 'm':
                self.update_speed("down")
            elif key_name == 'x':
                print("⏹ Stop")
                self.pressed_keys.clear()
            elif key_name == 'z':
                print("👋 Quitting...")
                self.running = False
        except AttributeError:
            pass
            
    def on_key_release(self, key):
        """Handle key release events."""
        try:
            key_name = key.name.lower()
            if key_name in ['w', 'a', 's', 'd', 'q', 'e']:
                self.pressed_keys.discard(key_name)
        except AttributeError:
            pass
        
    def control_loop(self):
        """Main control loop."""
        print("Starting control loop...")
        
        while self.running:
            try:
                # Update action based on pressed keys
                self.update_action_from_keys()
                
                # Send current action to robot
                if any(abs(v) > 0.001 for v in self.current_action.values()):
                    self.robot.send_action(self.current_action)
                else:
                    # Send stop command
                    stop_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                    self.robot.send_action(stop_action)
                
                time.sleep(0.05)  # 20 Hz control loop for better responsiveness
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error in control loop: {e}")
                break
                
    def run(self):
        """Run the teleoperation."""
        try:
            self.connect()
            self.print_controls()
            
            # Simple input mode
            print("📝 Enter-based input mode (press Enter after each key)")
            
            # Start control loop in separate thread
            self.running = True
            control_thread = threading.Thread(target=self.control_loop)
            control_thread.daemon = True
            control_thread.start()
            
            # Main input loop
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                char = sys.stdin.read(1)
                if char == 'z':
                    self.running = False
                    break
                elif char == 'n':
                    self.update_speed("up")
                elif char == 'm':
                    self.update_speed("down")
                elif char == 'x':
                    self.pressed_keys.clear()
                elif char in ['w', 'a', 's', 'd', 'q', 'e']:
                    self.pressed_keys.add(char)
                elif char == '\x1b':  # ESC key
                    self.pressed_keys.clear()
            
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.running = False
            self.disconnect()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="XLerobot Wheels Teleoperation")
    parser.add_argument("--port1", default="/dev/left-hand", help="Left hand port")
    parser.add_argument("--port2", default="/dev/right-hand", help="Right hand port") 
    parser.add_argument("--port3", default="/dev/neck", help="Neck port")
    parser.add_argument("--port4", default="/dev/wheels", help="Wheels port")
    parser.add_argument("--robot-id", default="xlerobot", help="Robot ID")
    
    args = parser.parse_args()
    
    # Create robot config
    config = XLerobotConfig(
        port1=args.port1,
        port2=args.port2, 
        port3=args.port3,
        port4=args.port4,
        id=args.robot_id
    )
    
    # Run teleoperation
    teleop = SimpleKeyboardWheelsTeleop(config)
    teleop.run()


if __name__ == "__main__":
    main()
