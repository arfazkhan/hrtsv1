
# H.E.A.R.T.S - Healthcare Enhanced by Artificial Intelligence and Realtime Tracking System

## Overview

**H.E.A.R.T.S** is an innovative healthcare technology project that integrates **facial recognition**, **Raspberry Pi**, and **real-time tracking** to streamline patient information management and improve workflow efficiency in healthcare settings. By leveraging advanced algorithms and augmented reality (AR) displays, H.E.A.R.T.S aims to reduce errors during patient handovers—especially during shift changes—and enhance decision-making for healthcare professionals.

---

## Key Features

- **Facial Recognition for Instant Patient Identification**
  - Uses **Histogram of Oriented Gradients (HOG)** for feature extraction.
  - Employs **Support Vector Machine (SVM)** for accurate classification.
  - Integrates **OpenCV** and **Dlib** for real-time image processing.

- **AR-Enabled Output Using Smart Glasses**
  - Displays patient details (name, age, blood group, medical history) on a microdisplay (FLCOS).
  - Provides a hands-free experience for healthcare professionals.

- **Interactive Controls**
  - **Capacitive Touch Sensor:** For intuitive on-device interactions.
  - **Push Button:** For quick navigation between information screens.

- **Real-Time Processing & Decision Support**
  - Retrieves and displays patient data instantly.
  - Minimizes human errors and miscommunications during shift changes.

---

## System Architecture

### Hardware Components

- **Raspberry Pi 4 B:**  
  Acts as the central processing unit.

- **Webcam:**  
  Captures real-time images for facial recognition.

- **Micro Display Module (FLCOS Display):**  
  Provides an augmented reality display for patient information.

- **Capacitive Touch Sensor & Push Button:**  
  Enable interactive user control.

### Software Components

- **Facial Recognition Algorithm:**  
  Combines HOG and SVM techniques for real-time facial identification.

- **Python Ecosystem:**
  - **OpenCV:** For image processing and face detection.
  - **Dlib:** For facial feature extraction and recognition.

- **Database Matching Module:**  
  Retrieves patient records based on the unique ID generated from facial recognition.

### Workflow

1. **Image Acquisition:**  
   The webcam captures the patient's image in real-time.

2. **Face Detection:**  
   OpenCV and Dlib detect the presence of a face.

3. **Feature Extraction:**  
   HOG is applied to extract facial features.

4. **Face Matching:**  
   An SVM classifier matches the extracted features to stored data, assigning a unique patient ID.

5. **AR Display:**  
   Patient information is retrieved from the database and displayed on the microdisplay for immediate access.

---

## Advantages

- **Enhanced Patient Safety:**  
  Ensures accurate patient identification and minimizes errors.

- **Improved Workflow Efficiency:**  
  Facilitates swift information transfer during shift changes.

- **Cost-Effective & Scalable:**  
  Utilizes affordable hardware and open-source software suitable for various healthcare settings.

- **Better Doctor-Patient Interaction:**  
  Hands-free AR displays allow healthcare professionals to stay focused on patient care.

---

## Future Scope

- **Integration with IoT Devices:**  
  Connect with wearable sensors, smart beds, etc.

- **Cloud & Mobile Connectivity:**  
  Enable remote access to patient data.

- **AI-Based Predictive Analytics:**  
  Analyze patient data to predict health trends.

- **Blockchain for Data Security:**  
  Enhance the security and integrity of patient data.

- **Voice Recognition:**  
  Implement voice commands for truly hands-free operation.

---

## Getting Started

### Prerequisites

- **Hardware:**
  - Raspberry Pi 4 B (or a compatible device)
  - Webcam (720p resolution or higher)
  - FLCOS Micro Display Module for AR
  - Capacitive Touch Sensor & Push Button

- **Software:**
  - Python
  - Required Python libraries: OpenCV, Dlib, NumPy, etc.
  - A configured database (e.g., SQLite, PostgreSQL) for patient records

### Installation

1. **Clone the Repository:**

   ```bash
   git clone[ https://github.com/yourusername/H.E.A.R.T.S.git](https://github.com/arfazkhan/hrtsv1)
   cd hrtsv1
   ```

2. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Database:**

   Set up your patient records database and update the configuration files as needed.

### Running the Application

Execute the main application with:

```bash
python app.py
```

---

## Development & Testing

- **Environment:**  
  Recommended to use a Linux-based OS (e.g., Raspbian) on Raspberry Pi.

- **Tools:**  
  Visual Studio Code, Git for version control, and Python virtual environments for dependency management.

- **Testing:**  
  - Unit tests for individual modules.
  - Integration tests to ensure proper hardware-software interactions.
  - Field testing in simulated clinical environments.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or further information, please contact:  
**Arfaz Khane** – [arfuzkhan@gmail.com](mailto:arfuzkhan@gmail.com)

---
