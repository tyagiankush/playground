import tkinter as tk


def check_code():
	"""Check if the user's code correctly sets the destination to Mars."""
	user_code = input_box.get('1.0', tk.END).strip()  # Get text from input box
	correct_code = 'destination = "Mars"\nprint(destination)'

	if user_code == correct_code:
		output_label.config(
			text='🚀 Course set for Mars!\nNavigation updated: Destination - Mars',
			fg='green',
		)
	else:
		output_label.config(
			text="❌ Oops! Something's not right.\nHint: Use destination = 'Mars' and print(destination)",
			fg='red',
		)


# Create main window
root = tk.Tk()
root.title('CodeQuest: Set Course for Mars')
root.geometry('600x400')
root.configure(bg='black')

# Title Label
title_label = tk.Label(
	root,
	text='🛰️ Set Course for Mars',
	font=('Arial', 16, 'bold'),
	fg='white',
	bg='black',
)
title_label.pack(pady=10)

# Instructions
instruction_label = tk.Label(
	root,
	text='Enter the correct code to set the destination to "Mars".',
	font=('Arial', 12),
	fg='white',
	bg='black',
)
instruction_label.pack()

# Code Input Box
input_box = tk.Text(root, height=5, width=50, font=('Courier', 12))
input_box.pack(pady=10)

# Run Button
run_button = tk.Button(
	root,
	text='Run Code',
	font=('Arial', 12, 'bold'),
	command=check_code,
	bg='blue',
	fg='white',
)
run_button.pack(pady=5)

# Output Label
output_label = tk.Label(root, text='', font=('Arial', 12), fg='white', bg='black')
output_label.pack(pady=10)

# Run the application
root.mainloop()
