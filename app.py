
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import time
import csv

# ---------------- Sidebar ----------------

st.sidebar.title("Traffic Monitoring System")

st.sidebar.markdown("""
### Features

- Vehicle Detection
- Vehicle Tracking
- Vehicle Counting
- Speed Estimation
- Traffic Density Analysis
- CSV Logging
""")

# ---------------- Title ----------------

st.title("Intelligent Traffic Monitoring System")

# ---------------- Upload Video ----------------

uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    with open(
        "uploaded_video.mp4",
        "wb"
    ) as f:

        f.write(uploaded_file.read())

    st.success("Video uploaded successfully!")

# ---------------- Load Model ----------------

model = YOLO("yolov8n.pt")

# ---------------- Process Button ----------------

if st.button("Process Video"):

    st.info("Processing video...")
    

    cap = cv2.VideoCapture("uploaded_video.mp4")
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ) 
    progress_bar = st.progress(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        "tracked_output.mp4",
        fourcc,
        fps,
        (width, height)
    )

    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    previous_positions = {}

    counted_ids = set()

    car_count = 0
    truck_count = 0
    bus_count = 0
    motorcycle_count = 0

    track_history = defaultdict(list)

    prev_time = 0

    speed_history = {}

    csv_file = open(
        "traffic_data.csv",
        mode="w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "Frame",
            "Visible Vehicles",
            "Cars",
            "Trucks",
            "Buses",
            "Motorcycles",
            "Traffic Status"
        ]
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break
        
        frame_number = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )
        
        progress = frame_number / total_frames
        
        progress_bar.progress(
            min(progress,1.0)
        )

        results = model(
            frame,
            classes=[2,3,5,7],
            verbose=False
        )

        detections = sv.Detections.from_ultralytics(
            results[0]
        )

        detections = tracker.update_with_detections(
            detections
        )

        current_vehicle_count = len(detections)

        labels = []

        for tracker_id, class_id, xyxy in zip(
            detections.tracker_id,
            detections.class_id,
            detections.xyxy
        ):

            if tracker_id is None:
                continue

            class_name = model.names[class_id]

            labels.append(
                f"{class_name} ID:{tracker_id}"
            )

            x1, y1, x2, y2 = xyxy

            center_x = int((x1+x2)/2)
            center_y = int((y1+y2)/2)

            cv2.circle(
                frame,
                (center_x, center_y),
                8,
                (255,0,0),
                -1
            )

            track = track_history[tracker_id]

            track.append(
                (center_x, center_y)
            )

            if len(track) > 30:
                track.pop(0)

            for j in range(1,len(track)):

                cv2.line(
                    frame,
                    track[j-1],
                    track[j],
                    (255,255,0),
                    3
                )

            if tracker_id in previous_positions:

                previous_x = previous_positions[
                    tracker_id
                ]

                crossed_line = (
                    (previous_x < 1000 and center_x >= 1000)
                    or
                    (previous_x > 1000 and center_x <= 1000)
                )

                if crossed_line and tracker_id not in counted_ids:

                    counted_ids.add(
                        tracker_id
                    )

                    class_name = class_name.strip().lower()

                    if class_name == "car":
                        car_count += 1

                    elif class_name == "truck":
                        truck_count += 1

                    elif class_name == "bus":
                        bus_count += 1

                    elif class_name == "motorcycle":
                        motorcycle_count += 1

            previous_positions[
                tracker_id
            ] = center_x

        if current_vehicle_count < 10:

            traffic_status = "LOW TRAFFIC"

        elif current_vehicle_count < 20:

            traffic_status = "MEDIUM TRAFFIC"

        else:

            traffic_status = "HIGH TRAFFIC"

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.line(
            frame,
            (1000,0),
            (1000,height),
            (0,0,255),
            8
        )

        cv2.putText(
            frame,
            f"Cars: {car_count}",
            (50,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        cv2.putText(
            frame,
            f"Trucks: {truck_count}",
            (50,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        cv2.putText(
            frame,
            f"Buses: {bus_count}",
            (50,170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        cv2.putText(
            frame,
            f"Motorcycles: {motorcycle_count}",
            (50,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        out.write(frame)

        csv_writer.writerow(
            [
                int(
                    cap.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                ),
                current_vehicle_count,
                car_count,
                truck_count,
                bus_count,
                motorcycle_count,
                traffic_status
            ]
        )

    cap.release()

    out.release()

    csv_file.close()

    st.success(
        "Processing Completed!"
    )
    # ---------------- Read CSV ----------------
    
    df = pd.read_csv(
        "traffic_data.csv"
    )
    
    latest_status = df[
        "Traffic Status"
    ].iloc[-1]
    
    if latest_status == "LOW TRAFFIC":
    
        st.success(
            latest_status
        )
    
    elif latest_status == "MEDIUM TRAFFIC":
    
        st.warning(
            latest_status
        )
    
    else:
    
        st.error(
            latest_status
        )


    
    # ---------------- Statistics ----------------
    
    st.subheader(
        "Overall Statistics"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Cars",
            int(df["Cars"].max())
        )
    
    with col2:
        st.metric(
            "Trucks",
            int(df["Trucks"].max())
        )
    
    with col3:
        st.metric(
            "Buses",
            int(df["Buses"].max())
        )
    
    with col4:
        st.metric(
            "Motorcycles",
            int(df["Motorcycles"].max())
        )
    
    # ---------------- Table ----------------
    
    st.subheader(
        "Traffic Statistics"
    )
    
    st.dataframe(
        df
    )
        
    st.subheader(
        "Vehicle Distribution"
    )
    
    fig2, ax2 = plt.subplots()
    
    ax2.pie(
    
        [
            df["Cars"].max(),
            df["Trucks"].max(),
            df["Buses"].max(),
            df["Motorcycles"].max()
        ],
    
        labels=[
            "Cars",
            "Trucks",
            "Buses",
            "Motorcycles"
        ],
    
        autopct="%1.1f%%"
    )
    
    st.pyplot(
        fig2
    )
    st.subheader(
        "Vehicle Count Comparison"
    )
    
    fig3, ax3 = plt.subplots()
    
    vehicle_names = [
        "Cars",
        "Trucks",
        "Buses",
        "Motorcycles"
    ]
    
    vehicle_counts = [
    
        df["Cars"].max(),
    
        df["Trucks"].max(),
    
        df["Buses"].max(),
    
        df["Motorcycles"].max()
    
    ]
    
    ax3.bar(
    
        vehicle_names,
    
        vehicle_counts
    
    )
    
    st.pyplot(
        fig3
    )
    st.subheader(
        "Traffic Volume Over Time"
    )
    
    fig, ax = plt.subplots(
        figsize=(10,5)
    )
    
    ax.plot(
        df["Frame"],
        df["Visible Vehicles"]
    )
    
    ax.set_xlabel(
        "Frame"
    )
    
    ax.set_ylabel(
        "Visible Vehicles"
    )
    
    ax.grid()
    
    st.pyplot(
        fig
    )
    st.subheader(
        "Processed Video"
    )
    
    st.video(
        "streamlit_video.mp4"
    )
    st.subheader(
        "Downloads"
    )
    
    with open(
        "traffic_data.csv",
        "rb"
    ) as file:
    
        st.download_button(
            "Download CSV Report",
            file,
            file_name="traffic_data.csv"
        )
    
    with open(
        "tracked_output.mp4",
        "rb"
    ) as file:
    
        st.download_button(
            "Download Processed Video",
            file,
            file_name="processed_video.mp4"
        )
    

