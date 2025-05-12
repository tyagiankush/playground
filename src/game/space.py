import logging
import tkinter as tk
from dataclasses import dataclass
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GameState:
	"""Model class representing the game state."""

	user_code: str = ""
	is_correct: bool = False
	message: str = ""
	score: int = 0
	current_level: int = 1


class GameController:
	"""Controller class handling game logic."""

	LEVELS = {
		1: {
			"correct_code": 'destination = "Mars"\nprint(destination)',
			"success_message": "🚀 Course set for Mars!\nNavigation updated: Destination - Mars",
			"hint": "Hint: Use destination = 'Mars' and print(destination)",
			"score": 100
		},
		2: {
			"correct_code": 'distance = 225000000  # km to Mars\nspeed = 50000  # km/h\ntravel_time = distance / speed\nprint(f"Travel time to Mars: {travel_time} hours")',
			"success_message": "⏱️ Perfect! You've calculated the travel time to Mars!\nThe journey will take 4500 hours (about 187.5 days)",
			"hint": "Hint: Calculate travel time using distance (225,000,000 km) divided by speed (50,000 km/h)",
			"score": 200
		}
	}
	MAX_LEVEL = 2

	def __init__(self, model: GameState):
		self.model = model

	def check_code(self, code: str) -> None:
		"""Check if the user's code is correct and update the game state."""
		try:
			self.model.user_code = code.strip()
			current_level_data = self.LEVELS[self.model.current_level]
			self.model.is_correct = code.strip() == current_level_data["correct_code"]

			if self.model.is_correct:
				self.model.message = current_level_data["success_message"]
				self.model.score += current_level_data["score"]
				if self.model.current_level < self.MAX_LEVEL:
					self.model.current_level += 1
					self.model.message += f"\n\n🎉 Level {self.model.current_level} unlocked!"
			else:
				self.model.message = f"❌ Oops! Something's not right.\n{current_level_data['hint']}"

			logger.info(f"Code checked. Correct: {self.model.is_correct}, Level: {self.model.current_level}")
		except Exception as e:
			logger.error(f"Error checking code: {str(e)}")
			self.model.message = "An error occurred while checking your code."


class GameView:
	"""View class handling the game UI."""

	def __init__(self, root: tk.Tk, controller: GameController):
		self.root = root
		self.controller = controller
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Set up the game UI components."""
		try:
			self.root.title("CodeQuest: Space Adventure")
			self.root.geometry("800x600")
			self.root.configure(bg="black")

			# Title Label
			self.title_label = tk.Label(
				self.root,
				text="🛰️ Space Code Adventure",
				font=("Arial", 16, "bold"),
				fg="white",
				bg="black",
			)
			self.title_label.pack(pady=10)

			# Level Label
			self.level_label = tk.Label(
				self.root,
				text="Level 1",
				font=("Arial", 12, "bold"),
				fg="yellow",
				bg="black",
			)
			self.level_label.pack()

			# Instructions
			self.instruction_label = tk.Label(
				self.root,
				text='Level 1: Enter the correct code to set the destination to "Mars".',
				font=("Arial", 12),
				fg="white",
				bg="black",
				wraplength=700,
			)
			self.instruction_label.pack()

			# Code Input Box
			self.input_box = tk.Text(self.root, height=8, width=70, font=("Courier", 12))
			self.input_box.pack(pady=10)

			# Run Button
			self.run_button = tk.Button(
				self.root,
				text="Run Code",
				font=("Arial", 12, "bold"),
				command=self.on_run_button_click,
				bg="blue",
				fg="white",
			)
			self.run_button.pack(pady=5)

			# Output Label
			self.output_label = tk.Label(
				self.root,
				text="",
				font=("Arial", 12),
				fg="white",
				bg="black",
				wraplength=700,
			)
			self.output_label.pack(pady=10)

			# Score Label
			self.score_label = tk.Label(
				self.root,
				text="Score: 0",
				font=("Arial", 12),
				fg="white",
				bg="black",
			)
			self.score_label.pack(pady=5)

		except Exception as e:
			logger.error(f"Error setting up UI: {str(e)}")
			raise

	def on_run_button_click(self) -> None:
		"""Handle the run button click event."""
		try:
			code = self.input_box.get("1.0", tk.END)
			self.controller.check_code(code)
			self.update_ui()
		except Exception as e:
			logger.error(f"Error in run button click: {str(e)}")
			self.output_label.config(text="An error occurred. Please try again.", fg="red")

	def update_ui(self) -> None:
		"""Update the UI based on the current game state."""
		try:
			self.output_label.config(
				text=self.controller.model.message,
				fg="green" if self.controller.model.is_correct else "red"
			)
			self.score_label.config(text=f"Score: {self.controller.model.score}")
			self.level_label.config(text=f"Level {self.controller.model.current_level}")
			
			# Update instructions based on level
			if self.controller.model.current_level == 1:
				self.instruction_label.config(
					text='Level 1: Enter the correct code to set the destination to "Mars".'
				)
			elif self.controller.model.current_level == 2:
				self.instruction_label.config(
					text="Level 2: Calculate the travel time to Mars using the given distance and speed.\n"
					"Distance to Mars: 225,000,000 km\n"
					"Spacecraft speed: 50,000 km/h"
				)
		except Exception as e:
			logger.error(f"Error updating UI: {str(e)}")


def main() -> None:
	"""Main function to run the game."""
	try:
		root = tk.Tk()
		model = GameState()
		controller = GameController(model)
		GameView(root, controller)
		root.mainloop()
	except Exception as e:
		logger.error(f"Application error: {str(e)}")
		raise


if __name__ == "__main__":
	main()
