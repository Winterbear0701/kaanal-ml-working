# main.py
import os
from exercises.jump_tracker import run_jump_tracker
from exercises.crunches import run_crunches

def display_results(exercise_name, sets_data):
    """
    Displays the results of all sets for an exercise in a tabular format.
    """
    print("\n" + "="*40)
    print(f" SESSION SUMMARY: {exercise_name.upper()} ".center(40, "="))
    print("="*40)
    
    if exercise_name == 'Vertical Jump':
        print(f"{'Set':<5} | {'Reps':<5} | {'Avg Height (relative)':<25}")
        print("-"*40)
        for i, height_data in enumerate(sets_data):
            reps = len(height_data)
            avg_height = sum(height_data) / reps if reps > 0 else 0
            print(f"{i+1:<5} | {reps:<5} | {avg_height:<25.2f}")
    else: # For crunches and other rep-based exercises
        print(f"{'Set':<5} | {'Reps':<5}")
        print("-"*40)
        for i, reps in enumerate(sets_data):
            print(f"{i+1:<5} | {reps:<5}")
            
    print("="*40 + "\n")


def main_menu():
    """
    Displays the main menu and handles user interaction.
    """
    while True:
        # Clear terminal screen for a cleaner menu
        os.system('cls' if os.name == 'nt' else 'clear') 

        print("💪 Welcome to POSECOUNTER! 💪")
        print("\nPlease select an exercise:")
        print("  1. Vertical Jump")
        print("  2. Crunches")
        print("  3. Exit")
        
        choice = input("Enter your choice (1-3): ")

        exercise_function = None
        exercise_name = ""

        if choice == '1':
            exercise_function = run_jump_tracker
            exercise_name = "Vertical Jump"
        elif choice == '2':
            exercise_function = run_crunches
            exercise_name = "Crunches"
        elif choice == '3':
            print("Thank you for using POSECOUNTER. Goodbye! 👋")
            print("A Product of PoseFit.AI")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
            continue
        
        # --- Exercise Loop ---
        sets_data = []
        set_number = 1
        while True:
            print(f"\n--- Starting Set {set_number} for {exercise_name} ---")
            print("Press 'q' in the video window when you are finished with the set.")
            input("Press Enter to begin...")

            # Run the selected exercise and get the result
            result = exercise_function() 
            sets_data.append(result)
            
            # Ask to continue
            continue_choice = input("Do another set? (y/n): ").lower()
            if continue_choice == 'y':
                set_number += 1
                continue
            else:
                display_results(exercise_name, sets_data)
                input("Press Enter to return to the main menu...")
                break # Exit exercise loop, go back to main menu

if __name__ == '__main__':
    main_menu()