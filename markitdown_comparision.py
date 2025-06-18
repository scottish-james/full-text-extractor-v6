#!/usr/bin/env python3
"""
Simple PowerPoint to Markdown Converter using MarkItDown
A GUI application built with Tkinter for easy file conversion
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pathlib import Path
import warnings

# Suppress the pydub ffmpeg warning since we're only doing PowerPoint conversion
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

try:
    from markitdown import MarkItDown
except ImportError:
    messagebox.showerror(
        "Missing Dependency",
        "MarkItDown is not installed.\n\nPlease install it with:\npip install 'markitdown[all]'"
    )
    exit(1)


class PowerPointToMarkdownGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PowerPoint to Markdown Converter")
        self.root.geometry("800x600")

        # Initialize MarkItDown
        self.md_converter = MarkItDown()

        # Variables
        self.selected_file = tk.StringVar()
        self.output_content = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
        """Create the user interface"""

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # File selection section
        ttk.Label(main_frame, text="Select PowerPoint File:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10)
        )

        # File path entry
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)

        self.file_entry = ttk.Entry(file_frame, textvariable=self.selected_file, width=60)
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        ttk.Button(file_frame, text="Browse...", command=self.browse_file).grid(row=0, column=1)

        # Convert button
        self.convert_btn = ttk.Button(
            main_frame,
            text="Convert to Markdown",
            command=self.convert_file,
            style="Accent.TButton"
        )
        self.convert_btn.grid(row=1, column=3, padx=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=1, column=4, padx=(10, 0), sticky=(tk.W, tk.E))

        # Output section
        ttk.Label(main_frame, text="Markdown Output:", font=("Arial", 12, "bold")).grid(
            row=2, column=0, columnspan=5, sticky=tk.W, pady=(20, 5)
        )

        # Text area with scrollbars
        self.text_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Consolas", 10)
        )
        self.text_area.grid(row=3, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=5, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="Save as...", command=self.save_markdown).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar(value="Ready to convert PowerPoint files...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 0))

    def browse_file(self):
        """Open file dialog to select PowerPoint file"""
        filetypes = (
            ('PowerPoint files', '*.pptx *.ppt'),
            ('All files', '*.*')
        )

        filename = filedialog.askopenfilename(
            title='Select PowerPoint file',
            initialdir=os.getcwd(),
            filetypes=filetypes
        )

        if filename:
            self.selected_file.set(filename)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")

    def convert_file(self):
        """Convert the selected PowerPoint file to Markdown"""
        file_path = self.selected_file.get().strip()

        if not file_path:
            messagebox.showwarning("No File Selected", "Please select a PowerPoint file first.")
            return

        if not os.path.exists(file_path):
            messagebox.showerror("File Not Found", f"The file '{file_path}' does not exist.")
            return

        # Disable the convert button and start progress
        self.convert_btn.config(state='disabled')
        self.progress.start()
        self.status_var.set("Converting file...")

        # Run conversion in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._convert_worker, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _convert_worker(self, file_path):
        """Worker thread for file conversion"""
        try:
            # Convert the file
            result = self.md_converter.convert(file_path)
            markdown_content = result.text_content

            # Update UI in main thread
            self.root.after(0, self._conversion_complete, markdown_content, None)

        except Exception as e:
            # Handle errors in main thread
            self.root.after(0, self._conversion_complete, None, str(e))

    def _conversion_complete(self, content, error):
        """Handle conversion completion in main thread"""
        # Stop progress and re-enable button
        self.progress.stop()
        self.convert_btn.config(state='normal')

        if error:
            messagebox.showerror("Conversion Error", f"Failed to convert file:\n\n{error}")
            self.status_var.set("Conversion failed")
        else:
            # Display the markdown content
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, content)

            file_name = os.path.basename(self.selected_file.get())
            self.status_var.set(f"Successfully converted {file_name}")

            # Show success message
            messagebox.showinfo("Success", "PowerPoint file converted to Markdown successfully!")

    def save_markdown(self):
        """Save the markdown content to a file"""
        content = self.text_area.get(1.0, tk.END).strip()

        if not content:
            messagebox.showwarning("No Content", "No markdown content to save.")
            return

        # Suggest filename based on input file
        input_file = self.selected_file.get()
        if input_file:
            suggested_name = Path(input_file).stem + ".md"
        else:
            suggested_name = "converted.md"

        filename = filedialog.asksaveasfilename(
            title='Save Markdown file',
            initialname=suggested_name,
            defaultextension='.md',
            filetypes=[('Markdown files', '*.md'), ('All files', '*.*')]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.status_var.set(f"Saved: {os.path.basename(filename)}")
                messagebox.showinfo("Saved", f"Markdown saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n\n{e}")

    def clear_output(self):
        """Clear the output text area"""
        self.text_area.delete(1.0, tk.END)
        self.status_var.set("Output cleared")

    def copy_to_clipboard(self):
        """Copy markdown content to clipboard"""
        content = self.text_area.get(1.0, tk.END).strip()

        if not content:
            messagebox.showwarning("No Content", "No content to copy.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("Content copied to clipboard")
        messagebox.showinfo("Copied", "Markdown content copied to clipboard!")


def main():
    """Main function to run the application"""
    root = tk.Tk()

    # Set the application icon (optional)
    try:
        root.iconbitmap("icon.ico")  # Add your icon file if you have one
    except:
        pass  # Ignore if icon file doesn't exist

    # Create and run the application
    app = PowerPointToMarkdownGUI(root)

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()