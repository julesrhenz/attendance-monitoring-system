import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from ultralytics import YOLO
from datetime import datetime
import os 
from PIL import Image, ImageTk
import threading
import subprocess
import sys
import queue
import webbrowser
import psycopg2


yolo_model = YOLO("C:/Users/Jules/model.pt")

def detect_faces_yolo(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes
    return boxes.xyxy if boxes else []

def capture_images_for_student(name, cap, db_path="C:/Users/Jules/Desktop/reference faces"):
    if not cap.isOpened():  # Check if the camera is opened
        messagebox.showerror("Camera Error", "Camera not initialized properly.")
        return
    
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        messagebox.showerror("Capture Error", "Failed to capture image.")
        return
    
    save_dir = os.path.join(db_path, name)
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    max_images = 100

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_yolo(frame)

        if len(faces) > 0:
            for face in faces:
                x1, y1, x2, y2 = [int(coord) for coord in face]
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                img_path = os.path.join(save_dir, f"{name}_{count+1}.jpg")
                cv2.imwrite(img_path, face_crop)
                print(f"Saved cropped face to {img_path}")
                count += 1

                if count >= max_images:
                    break

            cv2.waitKey(500)

    messagebox.showinfo("Done", f"Captured {count} cropped face images for {name},\n\nPlease run 'add embeddings.py' to add the newly added student to the stored facial embeddings.")

# ---------------- PostgreSQL Config ----------------
def get_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="romeopogi",
        host="localhost",
        port="5432"
    )

def get_attendance_tables():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' AND tablename LIKE 'attendance_%' 
            ORDER BY tablename DESC;
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in results]
    except Exception as e:
        messagebox.showerror("DB Error", f"Could not fetch tables:\n{e}")
        return []

# ---------------- Output Redirect ----------------
class RedirectText:
    def __init__(self, text_widget, output_queue):
        self.text_widget = text_widget
        self.output_queue = output_queue

    def write(self, string):
        self.output_queue.put(string)

    def flush(self):
        pass

# ---------------- Main Tkinter App ----------------
original_name = None
original_pattern = None
original_status = None
original_percentage = None 
last_added_student = None

def create_tkinter_window_with_camera():
    # Color palette
    dark_green = "#1B4332"
    light_green = "#2D6A4F"
    accent_green = "#40916C"
    text_color = "#FFFFFF"
    entry_bg = "#D8F3DC"
    
    global original_name, original_pattern, original_status, original_percentage
    root = tk.Tk()
    root.title("Attendance Monitoring System")
    root.geometry("1700x800")

    paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
    paned_window.pack(fill=tk.BOTH, expand=True)

    output_queue = queue.Queue()

    # --- Left Panel ---
    left_panel = tk.Frame(paned_window)
    left_panel.grid_rowconfigure(0, weight=3)
    left_panel.grid_rowconfigure(1, weight=0)
    left_panel.grid_rowconfigure(2, weight=2)
    left_panel.grid_columnconfigure(0, weight=1)
    paned_window.add(left_panel, minsize=300)

    video_label = tk.Label(left_panel, bg="black")
    video_label.grid(row=0, column=0, sticky="nsew")

    def run_external_script():
        print("Running external script...")
        try:
            process = subprocess.Popen(
                ["python", "-u", "C:/Users/Jules/Desktop/Thesis Codes/cam-2.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8'
            )
            for stdout_line in iter(process.stdout.readline, ""):
                output_queue.put(stdout_line)
            for stderr_line in iter(process.stderr.readline, ""):
                output_queue.put(stderr_line)
            process.stdout.close()
            process.stderr.close()
            process.wait()
        except Exception as e:
            output_queue.put(f"Failed to run external script: {e}\n")
            root.after(0, lambda: messagebox.showerror("Script Error", f"Failed to run external script:\n{e}"))

    def open_website():
        webbrowser.open("https://sh1on78.github.io/Face-AMS/")

    #def open_firebase_database():
        #firebase_url = "https://console.firebase.google.com/u/5/project/face-ams-1fa9f/database/face-ams-1fa9f-default-rtdb/data/~2FAttendance_records"  # Realtime DB
        #webbrowser.open(firebase_url)    

    button_frame = tk.Frame(left_panel)
    button_frame.grid(row=1, column=0, sticky="ew", pady=5)
    tk.Button(button_frame, text="Run Script", command=lambda: threading.Thread(target=run_external_script).start(), bg="skyblue").grid(row=0, column=0, padx=5, sticky="ew")
    tk.Button(button_frame, text="Open Website", command=open_website, bg="lightcoral").grid(row=0, column=1, padx=5, sticky="ew")
    #tk.Button(button_frame, text="Open Firebase RTDB", command=open_firebase_database, bg="blue").grid(row=0, column=2, padx=10, sticky="ew")
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)

    terminal_output = tk.Text(left_panel, wrap="word")
    terminal_output.grid(row=2, column=0, sticky="nsew", columnspan=2)
    sys.stdout = RedirectText(terminal_output, output_queue)
    sys.stderr = RedirectText(terminal_output, output_queue)

    def update_terminal_output():
        while not output_queue.empty():
            output = output_queue.get_nowait()
            terminal_output.insert(tk.END, output)
            terminal_output.see(tk.END)
        root.after(100, update_terminal_output)

    root.after(100, update_terminal_output)

    right_panel = tk.Frame(paned_window)
    right_panel.grid_rowconfigure(1, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)
    paned_window.add(right_panel, minsize=400)

    crud_frame = tk.Frame(right_panel)
    crud_frame.grid(row=0, column=0, sticky="ew", pady=5)
    for i in range(12):
        crud_frame.grid_columnconfigure(i, weight=1)

    # Frame for buttons
    button_control_frame = tk.Frame(crud_frame, bg=dark_green)
    button_control_frame.grid(row=1, column=0, columnspan=12, sticky="ew", pady=2)
    for i in range(12):
        button_control_frame.grid_columnconfigure(i, weight=1)

    # Entry style for consistency
    entry_style = {"padx": 2, "pady": 2, "sticky": "ew"}

    # Table dropdown
    table_var = tk.StringVar()
    table_dropdown = ttk.Combobox(
        crud_frame, textvariable=table_var,
        values=get_attendance_tables(), state='readonly'
    )
    table_dropdown.grid(row=0, column=1, columnspan=1, **entry_style)
    table_dropdown.bind("<<ComboboxSelected>>", lambda e: load_table())

    # Entry fields
    name_entry = tk.Entry(crud_frame)
    name_entry.grid(row=0, column=3, columnspan=1, **entry_style)

    pattern_entry = tk.Entry(crud_frame)
    pattern_entry.grid(row=0, column=5, columnspan=1, **entry_style)

    status_entry = tk.Entry(crud_frame)
    status_entry.grid(row=0, column=7, columnspan=1, **entry_style)

    percentage_entry = tk.Entry(crud_frame)
    percentage_entry.grid(row=0, column=9, columnspan=1, **entry_style)

    # Labels
    tk.Label(crud_frame, text="Table:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    tk.Label(crud_frame, text="Name:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
    tk.Label(crud_frame, text="Pattern:").grid(row=0, column=4, sticky="e", padx=4, pady=2)
    tk.Label(crud_frame, text="Status:").grid(row=0, column=6, sticky="e", padx=4, pady=2)
    tk.Label(crud_frame, text="Attendance %:").grid(row=0, column=8, sticky="e", padx=4, pady=2)

    def get_selected_table():
        return table_var.get()

    def get_current_selection():
        global original_name, original_pattern, original_status, original_percentage
        selected = tree.selection()
        if selected:
            values = tree.item(selected[0])['values']
            original_name = values[0]
            original_pattern = str(values[1]).zfill(6)
            original_status = values[2]
            original_percentage = float(values[3])

    def load_table():
        for row in tree.get_children():
            tree.delete(row)
        table_name = get_selected_table()
        if not table_name:
            return
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"SELECT student_name, pattern, status, attendance_percentage FROM {table_name} ORDER BY student_name;")
            rows = cur.fetchall()
            for row in rows:
                student_name, pattern, status, percentage = row
                tree.insert('', tk.END, values=(student_name, str(pattern).zfill(6), status, percentage))
            cur.close()
            conn.close()
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def refresh_dropdown_and_table():
        table_dropdown['values'] = get_attendance_tables()
        if table_var.get() not in table_dropdown['values'] and table_dropdown['values']:
            table_var.set(table_dropdown['values'][0])
        load_table()

    def create_record():
        global last_added_student
        name = name_entry.get()
        pattern = pattern_entry.get().strip().zfill(6)
        if len(pattern) != 6 or not all(c in '01' for c in pattern):
            messagebox.showwarning("Input Error", "Pattern must be exactly 6 digits consisting only with 1s and 0s.")
            return
        status = status_entry.get().strip().lower()  # Normalize to lowercase first
        if not name or not pattern or not status:
            messagebox.showwarning("Input Error", "All fields are required.")
            return
        if status not in ["present", "absent", "late"]:
            messagebox.showwarning("Input Error", "Status must be either 'Present' or 'Absent' (case-insensitive).")
            return
        status = status.capitalize()  # Convert to "Present" or "Absent"
        percentage = percentage_entry.get()
        if percentage not in ["100", "85", "0"]:
            messagebox.showwarning("Input Error", "Attendance % must be 100%, 85%, or 0%.")
            return
        table_name = get_selected_table()
        if not table_name:
            return
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {table_name} (student_name, pattern, status, attendance_percentage) VALUES (%s, %s, %s, %s);",
                (name, pattern, status, percentage)
            )
            conn.commit()
            cur.close()
            conn.close()

            last_added_student = name

            load_table()
            name_entry.delete(0, tk.END)
            pattern_entry.delete(0, tk.END)
            status_entry.delete(0, tk.END)
            percentage_entry.delete(0, tk.END)

            # Prompt user to capture images
            result = messagebox.askyesno("Capture Face", f"Do you want to capture face images for {name} now?")
            if result:
                # Run the image capture in a separate thread to prevent GUI freezing
                threading.Thread(target=capture_images_for_student, args=(name, cap), daemon=True).start()
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def update_record():
        get_current_selection()
        new_name = name_entry.get()
        new_pattern = pattern_entry.get()
        new_status = status_entry.get().strip().lower()
        if new_status not in ["present", "absent", "late"]:
            messagebox.showwarning("Input Error", "Status must be either 'Present' or 'Absent' (case-insensitive).")
            return
        new_status = new_status.capitalize()
        new_percentage = percentage_entry.get()
        if new_percentage not in ["100", "85", "0"]:
            messagebox.showwarning("Input Error", "Attendance % must be 100%, 85%, or 0%.")
            return
        if not new_name or not new_pattern or not new_status or not new_percentage:
            messagebox.showwarning("Input Error", "All fields are required.")
            return
        table_name = get_selected_table()
        if not table_name:
            return
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                UPDATE {table_name}
                SET student_name = %s, pattern = %s, status = %s, attendance_percentage = %s
                WHERE student_name = %s AND pattern = %s AND status = %s AND attendance_percentage = %s;
            """, (new_name, new_pattern, new_status, new_percentage, original_name, original_pattern, original_status, original_percentage))
            conn.commit()
            cur.close()
            conn.close()
            load_table()
            tree.selection_remove(tree.selection())
            name_entry.delete(0, tk.END)
            pattern_entry.delete(0, tk.END)
            status_entry.delete(0, tk.END)
            percentage_entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def delete_record():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Select a record to delete.")
            return

        values = tree.item(selected[0])['values']
        if not values or len(values) != 4:
            messagebox.showerror("Selection Error", "Invalid record selected.")
            return

        name, pattern, status, percentage = values
        table_name = get_selected_table()
        if not table_name:
            return
        
        print(f"Deleting record: Name={name}, Pattern={pattern}, Status={status}, Attendance %={percentage}")

        name = str(values[0]).strip()
        pattern = str(values[1]).strip().zfill(6)
        status = str(values[2]).strip()
        percentage = str(values[3]).strip()

        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete:\n\nName: {name}\nPattern: {pattern}\nStatus: {status}\nAttendance %: {percentage}")
        if not confirm:
            return

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                DELETE FROM {table_name} 
                WHERE student_name = %s AND pattern = %s AND status = %s AND attendance_percentage = %s;
            """, (name, str(pattern).zfill(6), status, percentage))
            conn.commit()

            if cur.rowcount == 0:
                messagebox.showinfo("Info", "No matching record found to delete.")
            else:
                messagebox.showinfo("Success", "Record deleted successfully.")
                load_table()

            cur.close()
            conn.close()
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def on_tree_select(event):
        get_current_selection()
        name_entry.delete(0, tk.END)
        pattern_entry.delete(0, tk.END)
        status_entry.delete(0, tk.END)
        percentage_entry.delete(0, tk.END)
        name_entry.insert(0, original_name)
        pattern_entry.insert(0, original_pattern)
        status_entry.insert(0, original_status)
        percentage_entry.insert(0, original_percentage)

    def capture_later():
        name = name_entry.get().strip()
        if not name:
            messagebox.showinfo("No Student", "No student name provided.")
            return

        result = messagebox.askyesno("Capture Face", f"Do you want to capture face images for {name} now?")
        if result:
            threading.Thread(target=capture_images_for_student, args=(name, cap), daemon=True).start()

    capture_button = tk.Button(button_control_frame,text="Capture Student", command=capture_later,bg="brown",width=15) 
    capture_button.grid(row=0, column=4, padx=2, pady=5)
    tk.Button(button_control_frame, text="Create", command=create_record, bg="lightgreen", width=15).grid(row=0, column=5, padx=2, pady=5)
    tk.Button(button_control_frame, text="Update", command=update_record, bg="lightyellow", width=15).grid(row=0, column=6, padx=2, pady=5)
    tk.Button(button_control_frame, text="Delete", command=delete_record, bg="tomato", width=15).grid(row=0, column=7, padx=2, pady=5)
    tk.Button(button_control_frame, text="Refresh", command=refresh_dropdown_and_table, bg="lightblue", width=15).grid(row=0, column=8, padx=2, pady=5)

    # Configure main window and panels
    root.configure(bg=dark_green)
    left_panel.configure(bg=dark_green)
    right_panel.configure(bg=dark_green)
    button_frame.configure(bg=dark_green)
    crud_frame.configure(bg=dark_green)

    # Configure video and terminal
    video_label.configure(bg="#000000")
    terminal_output.configure(bg="#081C15", fg="#B7E4C7", insertbackground="white")

    # Style buttons
    for btn in button_frame.winfo_children():
        if isinstance(btn, tk.Button):
            btn.configure(bg=accent_green, fg=text_color, font=('Arial', 10, 'bold'))

    # Style CRUD buttons and dropdown
    table_dropdown.configure(style='Custom.TCombobox')
    for widget in crud_frame.winfo_children():
        if isinstance(widget, tk.Label):
            widget.configure(bg=dark_green, fg=text_color, font=('Arial', 10, 'bold'))
        elif isinstance(widget, tk.Button):
            if widget["text"] == "Create":
                widget.configure(bg="#40916C", fg=text_color)
            elif widget["text"] == "Update":
                widget.configure(bg="#B7E4C7", fg="black")
            elif widget["text"] == "Delete":
                widget.configure(bg="#D00000", fg=text_color)
            elif widget["text"] == "Refresh":
                widget.configure(bg=accent_green, fg=text_color)

    # Style entry fields
    name_entry.configure(bg=entry_bg, fg="black")
    pattern_entry.configure(bg=entry_bg, fg="black")
    status_entry.configure(bg=entry_bg, fg="black")

    # Style the table frame and treeview
    table_frame = tk.Frame(right_panel, bg=dark_green)
    table_frame.grid(row=1, column=0, sticky="nsew")

    # Configure Treeview style with dark grid lines
    style = ttk.Style()
    style.configure("Custom.Treeview",
                    background=entry_bg,
                    foreground="black",
                    rowheight=25,
                    fieldbackground=entry_bg,
                    font=('Arial', 10))
    
    # Add dark borders/grid lines
    style.layout("Custom.Treeview", [
        ('Custom.Treeview.treearea', {'sticky': 'nswe', 'border': 1})
    ])
    style.configure("Custom.Treeview", 
                   borderwidth=1,
                   bordercolor='#1B4332',  # Dark green border
                   lightcolor='#2D6A4F',   # Slightly lighter green for grid
                   darkcolor='#1B4332')    # Dark green for grid
    
    # Configure dark heading style
    style.configure("Custom.Treeview.Heading",
                    background=light_green,
                    foreground=dark_green,
                    font=('Arial', 10, 'bold'),
                    borderwidth=1,
                    relief='raised')

    # Create and configure Treeview with alternating row colors
    tree = ttk.Treeview(table_frame, 
                        columns=("Student Name", "Pattern", "Status", "Attendance %"),
                        show="headings",
                        style="Custom.Treeview",
                        selectmode='browse')

    # Enable grid lines display
    tree.tag_configure('oddrow', background='#95D5B2')  # Lighter green for odd rows
    tree.tag_configure('evenrow', background=entry_bg)  # Regular background for even rows

    # Configure headings
    tree.heading("Student Name", text="Student Name")
    tree.heading("Pattern", text="Pattern")
    tree.heading("Status", text="Status")
    tree.heading("Attendance %", text ="Attendance %")

    # Configure column widths
    tree.column("Student Name", width=200, minwidth=150)
    tree.column("Pattern", anchor="center", width=100, minwidth=80)
    tree.column("Status", anchor="center", width=100, minwidth=80)
    tree.column("Attendance %", anchor="center", width=100, minwidth=80)

    tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    tree.bind('<<TreeviewSelect>>', on_tree_select)

    # Configure alternating row colors
    style.map('Custom.Treeview',
              background=[('selected', accent_green)],
              foreground=[('selected', text_color)])

    tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree_scroll.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=tree_scroll.set)
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    available_tables = get_attendance_tables()
    if available_tables:
        table_var.set(available_tables[0])
        load_table()

    cap = None
    for i in range(2):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"Camera {i} selected.")
            break
        temp_cap.release()

    if cap is None:
        print("Error: No available camera found.")
        video_label.config(text="Error: No available camera found.", fg="red")
        return

    def update_camera_feed():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Detect faces in the current frame
            faces = detect_faces_yolo(frame)

            # Draw bounding boxes around the detected faces
            for face in faces:
                x1, y1, x2, y2 = [int(coord) for coord in face]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for detected face

            video_label.update_idletasks()
            available_height = video_label.winfo_height()
            available_width = video_label.winfo_width()
            target_width = min(available_width, int(available_height * 4 / 3))
            target_height = int(target_width * 3 / 4)

            frame = cv2.resize(frame, (target_width, target_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            video_label.config(image=img)
            video_label.image = img

    threading.Thread(target=update_camera_feed, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    create_tkinter_window_with_camera()
