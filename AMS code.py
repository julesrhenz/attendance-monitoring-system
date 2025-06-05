import cv2
import os
import time
import csv
import traceback
from datetime import datetime, timedelta
from ultralytics import YOLO
from deepface import DeepFace
import psycopg2
from psycopg2 import sql  
import smtplib
import pickle
from numpy.linalg import norm
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from sklearn.metrics.pairwise import cosine_distances as cosine_dist
import tensorflow as tf
import warnings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load student emails from CSV
student_emails = {}
with open("C:/Users/Jules/Desktop/Thesis Codes/email.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        student_emails[row['student']] = row['email']

date_only = datetime.now().strftime("%Y-%m-%d")
monitoring_active = False
monitoring_should_stop = False
reference = "C:/Users/Jules/Desktop/reference faces"
padding = 20

rep_file = "C:/Users/Jules/representations_facenet_finalegit.pkl"
MODEL_NAME = "Facenet"
INPUT_SHAPE = (160, 160)
model_wrapper  = DeepFace.build_model(MODEL_NAME)
facenet_model = model_wrapper.model

LATE_PATTERNS = {
    "001111", "010111", "011011", "011101", "011110", "011111"
}
PRESENT_PATTERNS = {
    "100111", "101011", "101101", "101110", "101111",
    "110011", "110101", "110110", "110111", "111001",
    "111010", "111011", "111100", "111101", "111110", "111111"
}

def initialize_firebase():
    cred = credentials.Certificate("C:/Users/Jules/Documents/face-ams-1fa9f-firebase-adminsdk-fbsvc-c33d9c775b.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://face-ams-1fa9f-default-rtdb.asia-southeast1.firebasedatabase.app/"
    }) 

def initialize_fixed_attendance_table(reference):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="romeopogi",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fixed_attendance (
                student_name VARCHAR(100) PRIMARY KEY,
                capture_time TIMESTAMP,
                session_1 INTEGER DEFAULT 0,
                session_2 INTEGER DEFAULT 0,
                session_3 INTEGER DEFAULT 0,
                session_4 INTEGER DEFAULT 0,
                session_5 INTEGER DEFAULT 0,
                session_6 INTEGER DEFAULT 0,
                pattern VARCHAR(6) DEFAULT '000000',
                status VARCHAR(7) DEFAULT 'Absent',
                attendance_percentage FLOAT DEFAULT '0.0'
            );
        """)

        student_list = [name for name in os.listdir(reference) if os.path.isdir(os.path.join(reference, name))]

        for student in student_list:
            cur.execute("""
                INSERT INTO fixed_attendance (
                    student_name, capture_time, session_1, session_2,
                    session_3, session_4, session_5, session_6,
                    pattern, status, attendance_percentage
                ) VALUES (
                    %s, NULL, 0, 0, 0, 0, 0, 0, '000000', 'Absent', '0.0'
                )
                ON CONFLICT (student_name) DO NOTHING;
            """, (student,))

        conn.commit()

        # ✅ Optional logging to verify records are there
        cur.execute("SELECT COUNT(*) FROM fixed_attendance;")
        count = cur.fetchone()[0]
        print(f"📋 Initialized fixed_attendance table with {count} student(s).")

    except Exception as e:
        print(f"Error initializing the attendance table: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def attendance_trigger(event):
    global monitoring_active, monitoring_should_stop
    data = event.data
    is_monitoring_active = data.get("isMonitoringActive", False)

    if is_monitoring_active != monitoring_active:
        monitoring_active = is_monitoring_active
        if monitoring_active:
            print("✅ Firebase toggle detected: Starting attendance monitoring system...")
            monitoring_should_stop = False

            # Create new attendance table
            table_name = create_new_attendance_snapshot()

            capture_images(table_name=table_name)
        else:
            print("🔴 Firebase toggle detected: Attendance monitoring system deactivated. Waiting for next button press...")
            monitoring_should_stop = True

def create_new_attendance_snapshot():
    # NEW FUNCTION TO COPY fixed_attendance to a timestamped table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_name = f"attendance_{timestamp}"

    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="romeopogi",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute(sql.SQL("""
            CREATE TABLE {} AS
            SELECT * FROM fixed_attendance;
        """).format(sql.Identifier(table_name)))

        conn.commit()
        print(f"📁 Created snapshot table: {table_name}")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error creating snapshot table: {e}")

    return table_name

# 🔻 inside save_attendance_to_postgres:
def listen_for_attendance_trigger():
    ref = db.reference("systemControl/attendance")
    ref.listen(attendance_trigger)

def get_reference_faces(db_path):
    print(f"Loading reference faces from: {db_path}")
    reference_faces = {}
    for student_folder in os.listdir(db_path):
        student_folder_path = os.path.join(db_path, student_folder)
        if os.path.isdir(student_folder_path):
            reference_faces[student_folder] = {
                "name": student_folder,
                "pattern": "0" * 6
            }
            print(f"Added reference student: {student_folder}")
    print(f"Total reference students loaded: {len(reference_faces)}")
    return reference_faces

def preprocess_face(img):
    img = cv2.resize(img, INPUT_SHAPE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def cosine_dist(a, b):
        return 1 - np.dot(a, b) / (norm(a) * norm(b))

def recognize_face(img_path, rep_file, threshold=0.4):
    if not os.path.exists(rep_file):
        print("Embeddings not found. Please run build_face_database() first.")
        return None

    with open(rep_file, "rb") as f:
        representations = pickle.load(f)

    test_img = preprocess_face(img_path)
    test_embedding = facenet_model.predict(test_img)[0]

    best_distance = float("inf")
    best_match = None

    for rep in representations:
        dist = cosine_dist(rep["embedding"], test_embedding)
        if dist < best_distance:
            best_distance = dist
            best_match = rep["identity"]

    if best_distance < threshold:
        person = os.path.basename(os.path.dirname(best_match))
        print(f"Match: {person} (distance: {best_distance:.4f})")
        return person
    
    else:
        print("No match found.")
        return None

def log_attendance(run_folder, attendance_data, sessions, capture_timestamp, reference_students):
    sorted_students = sorted(reference_students)  # Ensure all students are considered
    print(f"Logging attendance for {len(sorted_students)} students")

    csv_file = os.path.join(run_folder, "attendance_log.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Student Name", "Capture Time"] + [f"Session {i+1}" for i in range(sessions)] + ["Status", "Attendance %"]
        writer.writerow(header)

        attendance_record = {}

        for student in sorted_students:
            # If student was never detected, ensure they still exist in the log
            if student not in attendance_data:
                attendance_data[student] = {"pattern": "000000"}  # Default pattern if not detected

            pattern = attendance_data.get(student, {}).get("pattern", "")  # Get the student's pattern

            # Determine status based on session pattern
            if pattern in PRESENT_PATTERNS:
                status = "Present"
                attendance = "100.0%"
            elif pattern in LATE_PATTERNS:
                status = "Late"
                attendance = "85.0%"
            else:
                status = "Absent"
                attendance = "0.0%"  # Default to absent if pattern remains all zeros

            # Write to CSV
            row = [student, capture_timestamp] + list(pattern) + [status, attendance]
            writer.writerow(row)

            # Update Firebase with attendance record
            attendance_record[student] = {
                "Status": status,
                "Sessions": pattern,
                "Attendance %": attendance
            }

    # Update Firebase with first date attendance
        db.reference(f"Attendance_records/{date_only}").set(attendance_record)
        print(f"Attendance log saved at: {csv_file} and uploaded to Firebase")

        # Create the second date's record and set initial status to absent (000000)
        second_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")  # +1 day from current
        second_attendance_record = {student: {"status": "Absent", "sessions": "000000"} for student in sorted_students}
        db.reference(f"Attendance_records/{second_date}").set(second_attendance_record)
        print(f"Second date attendance record for {second_date} added to Firebase")

        # Create the third date's record and set initial status to absent (000000)
        third_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")  # +2 days from current
        third_attendance_record = {student: {"status": "Absent", "sessions": "000000"} for student in sorted_students}
        db.reference(f"Attendance_records/{third_date}").set(third_attendance_record)
        print(f"Third date attendance record for {third_date} added to Firebase")

def save_attendance_to_postgres(attendance_data, capture_timestamp, repeat_sessions, table_name="fixed_attendance"):
    conn = None
    try:
        print("📝 Saving attendance to PostgreSQL...")
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="romeopogi",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                student_name VARCHAR(100) PRIMARY KEY,
                capture_time TIMESTAMP,
                session_1 INTEGER DEFAULT 0,
                session_2 INTEGER DEFAULT 0,
                session_3 INTEGER DEFAULT 0,
                session_4 INTEGER DEFAULT 0,
                session_5 INTEGER DEFAULT 0,
                session_6 INTEGER DEFAULT 0,
                pattern VARCHAR(6) DEFAULT '000000',
                status VARCHAR(7) DEFAULT 'Absent',
                attendance_percentage FLOAT DEFAULT '0.0'
            );
        """).format(sql.Identifier(table_name)))

        print("✅ Table ready or already exists.")

        for student, data in attendance_data.items():
            pattern = data["pattern"]
            session_flags = [int(c) for c in pattern]
            status = "Present" if pattern in PRESENT_PATTERNS else "Late" if pattern in LATE_PATTERNS else "Absent"
            attendance = "100.0" if pattern in PRESENT_PATTERNS else "85.0" if pattern in LATE_PATTERNS else "0.0"

            update_query = sql.SQL("""
                UPDATE {}
                SET capture_time = %s,
                    session_1 = %s,
                    session_2 = %s,
                    session_3 = %s,
                    session_4 = %s,
                    session_5 = %s,
                    session_6 = %s,
                    pattern = %s,
                    status = %s,
                    attendance_percentage = %s               
                WHERE student_name = %s;
            """).format(sql.Identifier(table_name))

            print(f"🛠 Updating record for: {student}")
            cur.execute(update_query, (
                capture_timestamp,
                *session_flags,
                pattern,
                status,
                attendance,
                student,
                
            ))

        conn.commit()
        print("✅ Attendance data successfully updated.")
        cur.close()

    except Exception as e:
        print("❌ Error saving to PostgreSQL:")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("🔒 PostgreSQL connection closed.")

def send_email_notification(student, status, to_email):
    try:
        from_email = "dennisraymundo53@gmail.com"
        subject = f"Attendance Status for {student}"
        body = f"Dear Parent/Guardian,\n\n{student}'s attendance status is: {status} for today's class.\n\nBest regards,\nAttendance System"

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, "zhyb auss jpdp ecpc")

        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        print(f"📧 Email sent to {student} ({to_email}) with status: {status}")
        server.quit()

    except Exception as e:
        print(f"❌ Error sending email to {student} ({to_email}): {e}")

def capture_images(
    images_per_student=5,
    repeat_sessions=6,
    session_duration_secs=120,
    model_path="C:/Users/Jules/model.pt",
    face_db="C:/Users/Jules/Desktop/reference faces",
    table_name="fixed_attendance"
):
    print("🎥 Starting camera-based attendance simulation...")

    # Setup
    run_name = "CPE103"
    run_name = re.sub(r'[<>:"/\\|?*]', "_", run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_name}_{timestamp}"
    output_folder = os.path.join("C:/Users/Jules/Desktop/Attendance Capture", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Check camera availability
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    cameras = []
    if cap1.isOpened():
        cameras.append(("Camera 1", cap1))
    if cap2.isOpened():
        cameras.append(("Camera 2", cap2))

    if not cameras:
        print("❌ Error: No available cameras found.")
        return

    print(f"✅ Cameras available: {[label for label, _ in cameras]}")
    cam_label, cap = cameras[0]  # Use the first available camera
    print(f"📷 Using {cam_label} for capturing.")

    model = YOLO(model_path)
    reference_students = get_reference_faces(face_db)
    attendance_data = {student: {"pattern": "000000"} for student in reference_students.keys()}
    session_image_counts = {student: [0] * repeat_sessions for student in reference_students}

    session_index = 0
    session_start_time = time.time()
    session_folder = os.path.join(output_folder, f"session_{session_index + 1}")
    os.makedirs(session_folder, exist_ok=True)
    print(f"\n▶️ Starting session {session_index + 1}...")

    student_face_count_per_session = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # Handle session timing
        elapsed = time.time() - session_start_time
        if elapsed >= session_duration_secs:
            session_index += 1
            if session_index >= repeat_sessions:
                print("✅ All sessions completed.")
                break

            session_start_time = time.time()
            session_folder = os.path.join(output_folder, f"session_{session_index + 1}")
            os.makedirs(session_folder, exist_ok=True)
            print(f"\n▶️ Starting session {session_index + 1}...")

            student_face_count_per_session = {}

        # Face detection
        results = model(frame)
        detected_faces = results[0].boxes
        print(f"Detected {len(detected_faces)} face(s)")

        for box in detected_faces:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_img = frame[y1:y2, x1:x2]

            if face_img.shape[0] < 100 or face_img.shape[1] < 100:
                print(f"⚠️ Skipping small face (size: {face_img.shape[1]}x{face_img.shape[0]})")
                continue

            label = recognize_face(face_img, rep_file, threshold=0.4)
            print(f"Recognition result: {label}")

            if label != "Unknown" and label is not None:
                if label not in attendance_data:
                    attendance_data[label] = {"pattern": "000000"}
                    session_image_counts[label] = [0] * repeat_sessions

                if label not in student_face_count_per_session:
                    student_face_count_per_session[label] = 0

                if student_face_count_per_session[label] < images_per_student:
                    save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                    face_path = os.path.join(session_folder, f"{label}_{save_time}.jpg")
                    cv2.imwrite(face_path, face_img)
                    student_face_count_per_session[label] += 1
                    session_image_counts[label][session_index] += 1
                    print(f"📸 Saved ({student_face_count_per_session[label]}/{images_per_student}) for {label}")
                else:
                    print(f"Skipping {label}, already has enough images.")
            else:
                print("Face could not be recognized.")

    cap.release()

    # Final attendance pattern generation
    for student in attendance_data:
        pattern = ""
        for i in range(repeat_sessions):
            session_folder_path = os.path.join(output_folder, f"session_{i+1}")
            session_files = os.listdir(session_folder_path)
            student_images = [f for f in session_files if f.startswith(student + "_") and f.endswith(".jpg")]
            pattern += "1" if len(student_images) >= images_per_student else "0"
        attendance_data[student]["pattern"] = pattern

    # Save to database and log
    capture_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_attendance_to_postgres(attendance_data, capture_timestamp, repeat_sessions, table_name)
    log_attendance(output_folder, attendance_data, repeat_sessions, capture_timestamp, list(reference_students.keys()))

    print("✅ Attendance logging completed.")
    print("📧 Sending email notifications...")

    for student in reference_students.keys():
        pattern = attendance_data.get(student, {}).get("pattern", "000000")
        if pattern in PRESENT_PATTERNS:
            status = "Present"
        elif pattern in LATE_PATTERNS:
            status = "Late"
        else:
            status = "Absent"

        to_email = student_emails.get(student)
        if to_email:
            send_email_notification(student, status, to_email)
        else:
            print(f"⚠️ No email found for {student}, skipping.")

    print("✅ All done.")


if __name__ == "__main__":
    initialize_fixed_attendance_table(reference)
    initialize_firebase()
    print("System is waiting for the button to activate monitoring...")
    listen_for_attendance_trigger()

    while True:
        if monitoring_should_stop:
            break
        time.sleep(1)
