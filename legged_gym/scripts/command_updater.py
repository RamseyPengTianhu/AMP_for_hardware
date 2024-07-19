import threading
import json

def get_command_input(prompt, default_value):
    """Get command input from user and return as a single float value."""
    user_input = input(prompt)
    if user_input.strip() == "":
        return default_value[0]  # Return the default value as a single float
    return float(user_input)

def update_command_range():
    """Continuously prompt the user for new command ranges."""
    default_ranges = {
        "lin_vel_x": [0.0, 0.0],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_yaw": [0.0, 0.0],
        "heading": [-1.0, 1.0]
    }
    
    while True:
        try:
            lin_vel_x = get_command_input("Enter new lin_vel_x value or press Enter to keep default: ", default_ranges["lin_vel_x"])
            lin_vel_y = get_command_input("Enter new lin_vel_y value or press Enter to keep default: ", default_ranges["lin_vel_y"])
            ang_vel_yaw = get_command_input("Enter new ang_vel_yaw value or press Enter to keep default: ", default_ranges["ang_vel_yaw"])
            heading = get_command_input("Enter new heading value or press Enter to keep default: ", default_ranges["heading"])

            command_interface = {
                "lin_vel_x": [lin_vel_x, lin_vel_x],
                "lin_vel_y": [lin_vel_y, lin_vel_y],
                "ang_vel_yaw": [ang_vel_yaw, ang_vel_yaw],
                "heading": [heading, heading]
            }
            
            with open('command_interface.json', 'w') as file:
                json.dump(command_interface, file)
            
            print(f"Updated command ranges to: {command_interface}")
        except ValueError as e:
            print("Invalid input. Please enter valid numerical values.")
            print(e)

if __name__ == "__main__":
    thread = threading.Thread(target=update_command_range)
    thread.start()