# Skincare and Beauty Analysis Web App

This is a Python-based web application that uses computer vision to provide skincare and beauty analysis.

## Features

- **Webcam Face Capture**: Captures the user's face using the webcam.
- **Skin Analysis**: Analyzes skin type, tone, and undertone.
- **Face Shape Detection**: Detects the user's face shape.
- **Product Recommendations**: Suggests skincare and makeup products.

## Project Structure

```
.
├── app.py
├── products.json
├── requirements.txt
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── script.js
└── templates
    └── index.html
```

## Setup and Installation

1.  **Clone the repository:**

    ```
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```
    pip install -r requirements.txt
    ```

4.  **Run the application:**

    ```
    python app.py
    ```

5.  Open your web browser and go to `http://127.0.0.1:5000`.

## How It Works

- The frontend is built with **HTML**, **CSS**, and **JavaScript**.
- **Bootstrap** is used for responsive UI components.
- **JavaScript** handles the webcam stream and captures a photo.
- The backend is a **Flask** application.
- **OpenCV** is used for image processing.
- **MediaPipe** is used for face landmark detection.
- Product recommendations are stored in a `products.json` file.
