import dataclasses
from queue import Queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox
import logging
import subprocess
from pathlib import Path
import sys
import threading
import colorama
from colorama import Fore


colorama.init(autoreset=True)

# Setup logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

CONVERSION_SCRIPT_FILE_PATH = Path(__file__).parent / "wdv3_jax.py"

@dataclasses.dataclass
class ThreadingData:
    file_path: str
    model: str
    prefix: str
    suffix: str
    blocked_tags: str
    is_multiple_models: bool
    save_captions_with_image: bool
    save_in_sub_folder: bool
    overwrite_files: bool


class ImageCaptioningGUI:
    """UI for the captioning"""

    def __init__(self, root_ui):
        """Constructor"""

        self.root_ui = root_ui
        root_ui.title("Image Captioning")

        # Use update_idletasks to ensure all layout calculations are complete
        root_ui.update_idletasks()
        root_ui.minsize(
            400, 440
        )

        # Prevent resizing the window to smaller than the minimum size
        root_ui.resizable(True, True)

        # Variables
        self.recursive_search = tk.BooleanVar(value=False)
        self.save_captions_with_image = tk.BooleanVar(value=True)
        self.save_captions_in_sub_folder = tk.BooleanVar(value=False)
        self.overwrite_files = tk.BooleanVar(value=True)
        self.model_vit = tk.BooleanVar(value=True)
        self.multi_threading = tk.BooleanVar(value=False)
        self.model_swinv2 = tk.BooleanVar(value=False)
        self.model_convnext = tk.BooleanVar(value=False)
        self.folder_path = tk.StringVar()
        self.prefix = tk.StringVar()
        self.suffix = tk.StringVar()
        self.blocked_tags = tk.StringVar()

        # Input Folder Section
        input_frame = tk.LabelFrame(root_ui, text="Input Folder")
        input_frame.grid(row=0, column=0, columnspan=3, sticky="we", padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.folder_path).grid(
            row=0, column=0, sticky="we", padx=5
        )
        tk.Button(input_frame, text="Browse", command=self.browse_folder).grid(
            row=0, column=1, padx=5
        )
        input_frame.grid_columnconfigure(0, weight=1)

        # Caption Settings Section
        options_frame = tk.LabelFrame(root_ui, text="Caption Settings")
        options_frame.grid(row=1, column=0, columnspan=3, sticky="we", padx=5, pady=5)
        tk.Label(options_frame, text="Prefix:").grid(row=0, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.prefix).grid(
            row=0, column=1, sticky="we"
        )
        tk.Label(options_frame, text="Suffix:").grid(row=1, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.suffix).grid(
            row=1, column=1, sticky="we"
        )
        tk.Label(options_frame, text="Blocked tags:").grid(row=2, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.blocked_tags).grid(
            row=2, column=1, sticky="we"
        )
        options_frame.grid_columnconfigure(1, weight=1)

        # Configuration Section
        config_frame = tk.LabelFrame(root_ui, text="Configuration")
        config_frame.grid(row=2, column=0, columnspan=3, sticky="we", padx=5, pady=5)
        tk.Checkbutton(
            config_frame,
            text="Multi-Threading (instead of multi-processing)",
            variable=self.multi_threading,
        ).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(
            config_frame, text="Recursive Search", variable=self.recursive_search
        ).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(
            config_frame, text="Overwrite existing files", variable=self.overwrite_files
        ).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(
            config_frame,
            text="Save captions with image",
            variable=self.save_captions_with_image,
        ).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(
            config_frame,
            text="Save captions in subfolder",
            variable=self.save_captions_in_sub_folder,
        ).grid(row=4, column=0, sticky="w")

        # Model Selection Section
        model_frame = tk.LabelFrame(root_ui, text="Select Model")
        model_frame.grid(row=3, column=0, columnspan=3, sticky="we", padx=5, pady=5)
        tk.Checkbutton(model_frame, text="VIT", variable=self.model_vit).grid(
            row=0, column=0, sticky="w"
        )
        tk.Checkbutton(model_frame, text="SwinV2", variable=self.model_swinv2).grid(
            row=1, column=0, sticky="w"
        )
        tk.Checkbutton(model_frame, text="ConvNext", variable=self.model_convnext).grid(
            row=2, column=0, sticky="w"
        )

        # Run Button
        run_button = tk.Button(root_ui, text="Run", command=self.run_captioning)
        run_button.grid(row=4, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        root_ui.grid_columnconfigure(0, weight=1)
        root_ui.grid_rowconfigure(2, weight=1)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            # Correctly update the StringVar linked to the Entry widget
            self.folder_path.set(
                folder_selected.replace("/", "\\")
            )  # Consistently use backslashes for Windows paths
            logging.debug(f"Folder selected: {self.folder_path.get()}")

    def run_captioning(self):
        folder = self.folder_path.get()
        logging.info(Fore.WHITE + f"Running captioning on folder: {folder}")
        logging.info(Fore.WHITE + f"Recursive search: {self.recursive_search.get()}")
        if not folder:
            messagebox.showerror("Error", "Please select a folder.")
            return

        models = self.get_selected_models()
        if not models:
            messagebox.showerror("Error", "Please select at least one model.")
            return

        # Early detection for skipping processing based on GUI settings
        if not self.save_captions_with_image.get() and not self.save_captions_in_sub_folder.get():
            logging.info(Fore.LIGHTYELLOW_EX + "Both saving options are disabled. Skipping processing of all files.")
            messagebox.showinfo("Info", "Both 'Save captions with image' and 'Save captions in subfolder' options are disabled. No files will be processed.")
            return

        files = self.get_files(folder, len(models))
        if not files:
            messagebox.showinfo("Info", "No image files found in the selected folder.")
            return

        process_queue = self.build_process_queue(models, files)
        if self.multi_threading.get():
            multi_threading_run(process_queue)
        else:
            multi_processing_run(process_queue)
        logging.info(Fore.CYAN + "Processing complete\n\n")

    def build_process_queue(self, models: list, file_paths: list):
        process_queue = []
        for model in models:
            for file_path in file_paths:
                threading_data = ThreadingData(
                    file_path=file_path,
                    model=model,
                    is_multiple_models=len(models) > 1,
                    save_captions_with_image=self.save_captions_with_image.get(),
                    save_in_sub_folder=self.save_captions_in_sub_folder.get(),
                    prefix=self.prefix.get(),
                    suffix=self.suffix.get(),
                    blocked_tags=self.blocked_tags.get(),
                    overwrite_files=self.overwrite_files.get()
                )
                process_queue.append(threading_data)
        return process_queue

    def get_selected_models(self):
        models = []
        if self.model_vit.get():
            models.append("vit")
        if self.model_swinv2.get():
            models.append("swinv2")
        if self.model_convnext.get():
            models.append("convnext")
        return models

    def get_files(self, folder: str, num_models: int):
        logging.info(f"Looking for files in: {folder}")
        extensions = ["*.jpg", "*.jpeg", "*.png"]
        files_grabbed = []
        for ext in extensions:
            # Use Path.rglob for recursive search or Path.glob for non-recursive
            if self.recursive_search.get():
                files_grabbed.extend(Path(folder).rglob(ext))
            else:
                files_grabbed.extend(Path(folder).glob(ext))

        # Convert Path objects to strings
        files_grabbed = [str(file) for file in files_grabbed]

        #logging.debug(f"Files found: {files_grabbed}")
        logging.info(Fore.YELLOW + f"{num_models} models selected.")
        logging.info(Fore.YELLOW + f"{len(files_grabbed)} image files found.")
        logging.info(
            Fore.YELLOW
            + f"Potentially running inference {num_models} * {len(files_grabbed)} = "
              f"{len(files_grabbed)*num_models} times.\n"
        )
        return files_grabbed


class ImageProcessingWorker(threading.Thread):

    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            data = self.queue.get()
            try:
                worker_process(data)
            finally:
                self.queue.task_done()


def worker_process(data):
    """This is the main worker process that creates txt files"""
    python_interpreter = sys.executable
    cmd = [
        python_interpreter,
        str(CONVERSION_SCRIPT_FILE_PATH),
        "--model",
        data.model,
        data.file_path,
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        output_lines = result.stdout.splitlines()

        caption = None
        for line in output_lines:
            if line.startswith("Caption:"):
                caption = line.split("Caption:")[1].strip()
                break

        if caption:
            # Apply prefix, suffix, and filter blocked tags
            prefix = data.prefix
            suffix = data.suffix
            blocked_tags = data.blocked_tags.split(",")
            blocked_tags = [tag.strip() for tag in blocked_tags if tag.strip()]
            caption_tags = caption.split(", ")
            filtered_tags = [tag for tag in caption_tags if tag not in blocked_tags]
            final_caption = ", ".join(filtered_tags).strip()

            # Adding prefix and suffix
            final_caption = f"{prefix} {final_caption} {suffix}".strip()

            # Cleanup: Remove any double spaces, trailing commas, and spaces
            final_caption = " ".join(final_caption.split()).replace(" ,", ",").rstrip(",")

            base_filename = Path(data.file_path).stem
            output_filename = f"{base_filename}.txt"

            if data.is_multiple_models or data.save_in_sub_folder:
                output_filename = f"{base_filename}.{data.model}.txt"

            if data.save_in_sub_folder:
                sub_folder_path = Path(data.file_path).parent / data.model
                sub_folder_path.mkdir(exist_ok=True)
                output_path = sub_folder_path / output_filename
            else:
                output_path = Path(data.file_path).with_name(output_filename)

            # Check if the file exists and if overwrite is disabled
            if output_path.exists() and not data.overwrite_files:
                logging.info(Fore.LIGHTYELLOW_EX + f"Skipping existing file: {output_path}")
                return  # Skip this file

            with open(output_path, "w") as f:
                f.write(final_caption)
            logging.info(Fore.GREEN + f"Caption saved: {output_path.name}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {data.file_path} with model {data.model}: {e.stderr}")


def multi_threading_run(threading_data: list[ThreadingData]):
    """Threading with standard Queue"""
    # Create a queue to communicate with the worker threads
    queue = Queue()
    # Create 8 worker threads
    for x in range(8):
        worker = ImageProcessingWorker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
    # Put the tasks into the queue as a tuple
    for one_thread_data in threading_data:
        logging.info("Queueing {}".format(one_thread_data.file_path))
        queue.put(one_thread_data)
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()


def multi_processing_run(threading_data: list[ThreadingData]):
    """Processing with Pool"""
    with ThreadPoolExecutor() as executor:
        executor.map(worker_process, threading_data, timeout=30)


def main():
    root = tk.Tk()
    app = ImageCaptioningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()