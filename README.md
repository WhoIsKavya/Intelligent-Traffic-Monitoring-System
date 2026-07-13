# 🚗 Intelligent Traffic Monitoring System

An AI-powered **Intelligent Traffic Monitoring System** built using **YOLOv8**, **ByteTrack**, **OpenCV**, and **Streamlit**. The application detects, tracks, and counts vehicles from uploaded traffic videos, analyzes traffic density, and provides an interactive dashboard with visual analytics.

---

## 📌 Project Overview

This project leverages state-of-the-art computer vision techniques to automate traffic monitoring. Users can upload a traffic video through a Streamlit web application, and the system processes the video to:

* Detect vehicles in each frame using YOLOv8.
* Track vehicles across frames using ByteTrack.
* Count vehicles crossing a virtual counting line.
* Estimate traffic density.
* Generate traffic statistics.
* Visualize results through an interactive dashboard.

---

## ✨ Features

* 🚗 Vehicle Detection using YOLOv8
* 🎯 Multi-Object Tracking using ByteTrack
* 🔢 Vehicle Counting
* 📈 Traffic Density Analysis
* 📊 Interactive Dashboard
* 📹 Upload and Process Custom Videos
* 📄 Automatic CSV Report Generation
* 📉 Traffic Volume Visualization
* 💾 Download Processed Video
* 📥 Download Traffic Statistics

---

## 🛠️ Tech Stack

| Category             | Technologies         |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Deep Learning        | YOLOv8 (Ultralytics) |
| Object Tracking      | ByteTrack            |
| Computer Vision      | OpenCV               |
| Dashboard            | Streamlit            |
| Data Analysis        | Pandas, NumPy        |
| Visualization        | Matplotlib           |
| Tracking Utilities   | Supervision          |

---

## 📂 Project Structure

```text
intelligent-traffic-monitoring-system/
│
├── app.py
├── vehicle_detection.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-traffic-monitoring-system.git
```

Navigate to the project directory:

```bash
cd intelligent-traffic-monitoring-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 📖 How to Use

1. Launch the Streamlit application.
2. Upload a traffic video (`.mp4`, `.avi`, or `.mov`).
3. Click **Process Video**.
4. Wait for the processing to complete.
5. View:

   * Vehicle counts
   * Traffic statistics
   * Traffic status
   * Traffic volume graph
   * Processed video
6. Download the processed video and CSV report if needed.

---

## 📊 Output

The application provides:

* Total Cars Detected
* Total Trucks Detected
* Total Buses Detected
* Total Motorcycles Detected
* Traffic Status (Low / Medium / High)
* Traffic Volume Graph
* Processed Video
* CSV Report

---

## 🚀 Future Improvements

* Vehicle Speed Estimation
* Lane-wise Vehicle Counting
* Automatic Traffic Violation Detection
* Real-time CCTV Stream Processing
* Emergency Vehicle Detection
* Interactive Dashboard Enhancements
* Database Integration
* Cloud Deployment

---

## 📚 Skills Demonstrated

* Computer Vision
* Deep Learning
* Object Detection
* Multi-Object Tracking
* Video Analytics
* Data Visualization
* Streamlit Application Development
* Python Programming

---

## 👩‍💻 Author

**Kavya Singh**

M.Tech (Computer Science) | AI & Machine Learning Enthusiast

GitHub: https://github.com/WhoIsKavya
