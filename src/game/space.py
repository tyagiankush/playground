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


class GameController:
	"""Controller class handling game logic."""

	CORRECT_CODE = 'destination = "Mars"\nprint(destination)'
	MAX_SCORE = 100

	def __init__(self, model: GameState):
		self.model = model

	def check_code(self, code: str) -> None:
		"""Check if the user's code is correct and update the game state."""
		try:
			self.model.user_code = code.strip()
			self.model.is_correct = code.strip() == self.CORRECT_CODE

			if self.model.is_correct:
				self.model.message = "🚀 Course set for Mars!\nNavigation updated: Destination - Mars"
				self.model.score = self.MAX_SCORE
			else:
				self.model.message = (
					"❌ Oops! Something's not right.\nHint: Use destination = 'Mars' and print(destination)"
				)
				self.model.score = 0

			logger.info(f"Code checked. Correct: {self.model.is_correct}")
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
			self.root.title("CodeQuest: Set Course for Mars")
			self.root.geometry("600x400")
			self.root.configure(bg="black")

			# Title Label
			self.title_label = tk.Label(
				self.root,
				text="🛰️ Set Course for Mars",
				font=("Arial", 16, "bold"),
				fg="white",
				bg="black",
			)
			self.title_label.pack(pady=10)

			# Instructions
			self.instruction_label = tk.Label(
				self.root,
				text='Enter the correct code to set the destination to "Mars".',
				font=("Arial", 12),
				fg="white",
				bg="black",
			)
			self.instruction_label.pack()

			# Code Input Box
			self.input_box = tk.Text(self.root, height=5, width=50, font=("Courier", 12))
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
			self.output_label = tk.Label(self.root, text="", font=("Arial", 12), fg="white", bg="black")
			self.output_label.pack(pady=10)

			# Score Label
			self.score_label = tk.Label(self.root, text="Score: 0", font=("Arial", 12), fg="white", bg="black")
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
				text=self.controller.model.message, fg="green" if self.controller.model.is_correct else "red"
			)
			self.score_label.config(text=f"Score: {self.controller.model.score}")
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
