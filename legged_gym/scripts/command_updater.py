import json
import os
from pynput import keyboard

# Define default command values
default_command_interface = {
    "lin_vel_x": [0.0, 0.0],
    "lin_vel_y": [0.0, 0.0],
    "ang_vel_yaw": [0.0, 0.0],
    "heading": [0.0, 0.0]
}

def save_to_json(filename, data):
    """Save the command interface dictionary to a JSON file."""
    try:
        with open(filename, 'w') as file:
            json.dump(data, file)
        print(f"Updated command ranges to: {data}")
    except IOError as e:
        print(f"Failed to write to {filename}: {e}")

def load_json(filename, default_data):
    """Load the JSON file, initializing it if it's empty or missing."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"'{filename}' is missing or empty. Initializing with default values.")
        save_to_json(filename, default_data)
    
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"'{filename}' contains invalid JSON. Reinitializing with default values.")
        save_to_json(filename, default_data)
        return default_data

def _check_command_interface():
    """Check and load the command interface from JSON, ensuring it is valid."""
    command_interface_file = 'command_interface.json'
    # Ensure command_interface.json is properly initialized and loaded
    return load_json(command_interface_file, default_command_interface)

# Initialize the command interface at the start
command_interface = _check_command_interface()

def get_command_input(prompt, default_value):
    """Get command input from user and return as a single float value."""
    user_input = input(prompt)
    if user_input.strip() == "":
        return default_value[0]
    return float(user_input)

def manual_input_mode():
    """Prompt the user for new command ranges in manual input mode."""
    default_ranges = {
        "lin_vel_x": [0.0, 0.0],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_yaw": [0.0, 0.0],
        "heading": [-1.0, 1.0]
    }
    
    print("Entering Manual Input Mode. Type 'm' to switch modes or 'q' to quit.")
    
    while True:
        try:
            lin_vel_x = get_command_input("Enter new lin_vel_x or press Enter to keep default: ", default_ranges["lin_vel_x"])
            lin_vel_y = get_command_input("Enter new lin_vel_y or press Enter to keep default: ", default_ranges["lin_vel_y"])
            ang_vel_yaw = get_command_input("Enter new ang_vel_yaw or press Enter to keep default: ", default_ranges["ang_vel_yaw"])
            heading = get_command_input("Enter new heading or press Enter to keep default: ", default_ranges["heading"])

            command_interface.update({
                "lin_vel_x": [lin_vel_x, lin_vel_x],
                "lin_vel_y": [lin_vel_y, lin_vel_y],
                "ang_vel_yaw": [ang_vel_yaw, ang_vel_yaw],
                "heading": [heading, heading]
            })
            
            save_to_json('command_interface.json', command_interface)

            choice = input("Press 'm' to switch modes or 'q' to quit: ").strip().lower()
            if choice == 'm':
                break
            elif choice == 'q':
                print("Exiting Manual Input Mode.")
                return 'quit'
            
        except ValueError:
            print("Invalid input. Please enter valid numerical values.")

def on_press(key):
    """Handle key press events to control commands or switch modes."""
    global command_interface
    increment_x = 0.5
    increment_yaw = 0.5

    try:
        if key == keyboard.Key.up:
            command_interface["lin_vel_x"][0] += increment_x
            command_interface["lin_vel_x"][1] += increment_x
            save_to_json('command_interface.json', command_interface)
        elif key == keyboard.Key.down:
            command_interface["lin_vel_x"][0] -= increment_x
            command_interface["lin_vel_x"][1] -= increment_x
            save_to_json('command_interface.json', command_interface)
        elif key == keyboard.Key.left:
            command_interface["ang_vel_yaw"][0] += increment_yaw
            command_interface["ang_vel_yaw"][1] += increment_yaw
            save_to_json('command_interface.json', command_interface)
        elif key == keyboard.Key.right:
            command_interface["ang_vel_yaw"][0] -= increment_yaw
            command_interface["ang_vel_yaw"][1] -= increment_yaw
            save_to_json('command_interface.json', command_interface)
        elif key.char == 'm':
            print("Switching to Manual Input Mode.")
            return False
        elif key.char == 'q':
            print("Exiting program.")
            return False
    except AttributeError:
        pass

def arrow_key_control_mode():
    """Listen to arrow key and mode-switching inputs."""
    print("Entering Arrow Key Control Mode. Use ↑/↓ for lin_vel_x, ←/→ for ang_vel_yaw.")
    print("Press 'm' to return to Manual Input Mode or 'q' to quit.")
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def choose_mode():
    """Prompt user to select a mode at the start or after each mode completes."""
    while True:
        print("\nSelect Mode:")
        print("1. Manual Input Mode")
        print("2. Arrow Key Control Mode")
        print("3. Quit")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            result = manual_input_mode()
            if result == 'quit':
                break
        elif choice == '2':
            arrow_key_control_mode()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    # Ensure that the JSON file is initialized and contains valid data before starting
    command_interface = _check_command_interface()
    choose_mode()
