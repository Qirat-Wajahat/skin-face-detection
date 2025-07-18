from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import base64

app = Flask(__name__)

# Load product recommendations
try:
    with open('products.json', 'r') as f:
        products = json.load(f)
except FileNotFoundError:
    products = {"skincare": {}, "makeup": {}}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read image file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        print("Starting analysis...")  # Debug log

        # --- Comprehensive Analysis Functions ---
        try:
            skin_type = analyze_skin_type(image)
            print(f"Skin type: {skin_type}")  # Debug log
        except Exception as e:
            print(f"Skin type analysis error: {e}")
            skin_type = "normal"
        
        try:
            skin_tone, undertone = analyze_skin_tone(image)
            print(f"Skin tone: {skin_tone}, Undertone: {undertone}")  # Debug log
        except Exception as e:
            print(f"Skin tone analysis error: {e}")
            skin_tone, undertone = "medium", "neutral"
        
        try:
            face_shape = analyze_face_shape(image)
            print(f"Face shape: {face_shape}")  # Debug log
        except Exception as e:
            print(f"Face shape analysis error: {e}")
            face_shape = "oval"
        
        try:
            eye_shape = analyze_eye_shape(image)
            print(f"Eye shape: {eye_shape}")  # Debug log
        except Exception as e:
            print(f"Eye shape analysis error: {e}")
            eye_shape = "almond"
        
        try:
            lip_shape = analyze_lip_shape(image)
            print(f"Lip shape: {lip_shape}")  # Debug log
        except Exception as e:
            print(f"Lip shape analysis error: {e}")
            lip_shape = "medium"
        
        try:
            nose_shape = analyze_nose_shape(image)
            print(f"Nose shape: {nose_shape}")  # Debug log
        except Exception as e:
            print(f"Nose shape analysis error: {e}")
            nose_shape = "medium"
        
        if face_shape == "No face detected":
            return jsonify({"error": "No face detected. Please try again."})

        print("Basic analysis completed, starting beauty analysis...")  # Debug log

        # --- Enhanced Beauty Analysis ---
        try:
            lip_analysis = analyze_lip_colors_for_skin_tone(skin_tone, undertone)
            print("Lip analysis completed")  # Debug log
        except Exception as e:
            print(f"Lip analysis error: {e}")
            lip_analysis = {
                "best_colors": ["Universal Nude", "Classic Red"],
                "avoid_colors": ["Extreme colors"],
                "technique": "Choose colors that complement your natural undertone.",
                "products": []
            }
        
        try:
            blush_analysis = analyze_blush_for_skin_tone(skin_tone, undertone)
            print("Blush analysis completed")  # Debug log
        except Exception as e:
            print(f"Blush analysis error: {e}")
            blush_analysis = {
                "best_shades": ["Universal Rose", "Soft Pink"],
                "avoid_shades": ["Extreme colors"],
                "application": "Apply to the apples of cheeks and blend upward.",
                "products": []
            }
        
        try:
            eye_makeup_analysis = analyze_eye_makeup_for_eye_shape(eye_shape)
            print("Eye makeup analysis completed")  # Debug log
        except Exception as e:
            print(f"Eye makeup analysis error: {e}")
            eye_makeup_analysis = {
                "eyeliner_style": "Classic liner along the upper lash line",
                "eyeshadow_technique": "Apply shadow in the crease and outer corner",
                "mascara_technique": "Apply evenly to all lashes",
                "avoid": "Overly dramatic looks",
                "technique_steps": ["Apply a neutral base shadow", "Use a medium shade in the crease"],
                "products": []
            }
        
        try:
            nose_contouring_analysis = analyze_nose_contouring_needs(nose_shape, face_shape)
            print("Nose contouring analysis completed")  # Debug log
        except Exception as e:
            print(f"Nose contouring analysis error: {e}")
            nose_contouring_analysis = {
                "needs_contouring": False,
                "contouring_areas": [],
                "technique": "Your nose has natural proportions.",
                "highlight_areas": ["Center of the nose bridge", "Tip of the nose"],
                "steps": ["Apply light highlight down the center of the nose", "Highlight the tip of the nose", "Blend for a subtle glow"],
                "products": [
                    {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                    {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
                ]
            }
        
        try:
            face_contouring_analysis = analyze_face_contouring_needs(face_shape)
            print("Face contouring analysis completed")  # Debug log
        except Exception as e:
            print(f"Face contouring analysis error: {e}")
            face_contouring_analysis = {
                "needs_contouring": False,
                "contouring_areas": [],
                "technique": "Your face has natural proportions.",
                "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
                "steps": ["Apply contour along the jawline", "Contour under the cheekbones", "Apply contour along the hairline", "Contour under the chin", "Highlight the center of forehead, under eyes, and chin", "Blend thoroughly for a natural look"],
                "products": [
                    {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                    {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
                ]
            }

        print("Beauty analysis completed, getting recommendations...")  # Debug log

        # Get comprehensive recommendations
        try:
            recommendations = get_comprehensive_recommendations(
                skin_type, skin_tone, undertone, face_shape, 
                eye_shape, lip_shape, nose_shape
            )
            print("Recommendations completed")  # Debug log
        except Exception as e:
            print(f"Recommendations error: {e}")
            recommendations = {
                "skincare": [],
                "makeup": [],
                "tips": []
            }

        print("Formatting results...")  # Debug log

        # Format results for frontend
        features = {
            "face_shape": {
                "value": face_shape,
                "description": f"Your face has a {face_shape.lower()} shape, which is perfect for specific makeup techniques."
            },
            "skin_tone": {
                "value": skin_tone,
                "description": f"Your skin tone is {skin_tone.lower()}, which works beautifully with warm and cool tones."
            },
            "skin_type": {
                "value": skin_type,
                "description": f"Your skin type is {skin_type.lower()}, requiring specific care and product recommendations."
            },
            "eye_shape": {
                "value": eye_shape,
                "description": f"Your {eye_shape.lower()} eyes can be enhanced with specific eyeliner and shadow techniques."
            },
            "lip_shape": {
                "value": lip_shape,
                "description": f"Your {lip_shape.lower()} lips can be beautifully defined with the right lip products."
            },
            "nose_shape": {
                "value": nose_shape,
                "description": f"Your {nose_shape.lower()} nose can be contoured to create perfect balance."
            }
        }

        # Enhanced beauty analysis results
        beauty_analysis = {
            "lip_colors": {
                "title": f"Perfect Lip Colors for {skin_tone} {undertone} Skin",
                "best_colors": lip_analysis["best_colors"],
                "avoid_colors": lip_analysis["avoid_colors"],
                "technique": lip_analysis["technique"],
                "products": lip_analysis["products"]
            },
            "blush_shades": {
                "title": f"Ideal Blush Shades for {skin_tone} {undertone} Skin",
                "best_shades": blush_analysis["best_shades"],
                "avoid_shades": blush_analysis["avoid_shades"],
                "application": blush_analysis["application"],
                "products": blush_analysis["products"]
            },
            "eye_makeup": {
                "title": f"Eye Makeup for {eye_shape} Eyes",
                "eyeliner_style": eye_makeup_analysis["eyeliner_style"],
                "eyeshadow_technique": eye_makeup_analysis["eyeshadow_technique"],
                "mascara_technique": eye_makeup_analysis["mascara_technique"],
                "avoid": eye_makeup_analysis["avoid"],
                "technique_steps": eye_makeup_analysis["technique_steps"],
                "products": eye_makeup_analysis["products"]
            },
            "nose_contouring": {
                "title": f"Nose Contouring for {nose_shape} Nose",
                "needs_contouring": nose_contouring_analysis["needs_contouring"],
                "contouring_areas": nose_contouring_analysis["contouring_areas"],
                "technique": nose_contouring_analysis["technique"],
                "highlight_areas": nose_contouring_analysis["highlight_areas"],
                "steps": nose_contouring_analysis["steps"],
                "products": nose_contouring_analysis["products"]
            },
            "face_contouring": {
                "title": f"Face Contouring for {face_shape} Face",
                "needs_contouring": face_contouring_analysis["needs_contouring"],
                "contouring_areas": face_contouring_analysis["contouring_areas"],
                "technique": face_contouring_analysis["technique"],
                "highlight_areas": face_contouring_analysis["highlight_areas"],
                "steps": face_contouring_analysis["steps"],
                "products": face_contouring_analysis["products"]
            }
        }

        print("Analysis completed successfully!")  # Debug log

        return jsonify({
            "features": features,
            "makeup_recommendations": recommendations.get("makeup", []),
            "skincare_recommendations": recommendations.get("skincare", []),
            "tips": recommendations.get("tips", []),
            "beauty_analysis": beauty_analysis
        })
        
    except Exception as e:
        print(f"Unexpected error in analyze route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/debug', methods=['POST'])
def debug_analysis():
    # Check if image file is in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    
    # Read image file
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Get detailed debug information
    debug_info = get_debug_info(image)
    
    return jsonify(debug_info)

def get_debug_info(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return {"error": "No face detected in debug mode"}
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        
        # Get landmark coordinates
        debug_data = {
            "image_size": f"{w}x{h}",
            "landmarks_detected": len(landmarks),
            "face_shape_analysis": {},
            "skin_tone_analysis": {},
            "eye_analysis": {},
            "lip_analysis": {},
            "nose_analysis": {}
        }
        
        try:
            # Face shape debug info
            forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
            chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            jaw_left = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            jaw_right = (int(landmarks[454].x * w), int(landmarks[454].y * h))
            cheek_left = (int(landmarks[132].x * w), int(landmarks[132].y * h))
            cheek_right = (int(landmarks[361].x * w), int(landmarks[361].y * h))
            
            face_width = np.linalg.norm(np.array(jaw_left) - np.array(jaw_right))
            face_height = chin[1] - forehead[1]
            cheek_width = np.linalg.norm(np.array(cheek_left) - np.array(cheek_right))
            
            height_width_ratio = face_height / face_width
            cheek_jaw_ratio = cheek_width / face_width
            
            debug_data["face_shape_analysis"] = {
                "forehead_coords": forehead,
                "chin_coords": chin,
                "jaw_left_coords": jaw_left,
                "jaw_right_coords": jaw_right,
                "cheek_left_coords": cheek_left,
                "cheek_right_coords": cheek_right,
                "face_width": round(face_width, 2),
                "face_height": round(face_height, 2),
                "cheek_width": round(cheek_width, 2),
                "height_width_ratio": round(height_width_ratio, 3),
                "cheek_jaw_ratio": round(cheek_jaw_ratio, 3)
            }
            
            # Skin tone debug info
            keypoints = [landmarks[10], landmarks[234], landmarks[454], landmarks[152], landmarks[132], landmarks[361]]
            color_samples = []
            
            for i, point in enumerate(keypoints):
                x = int(point.x * w)
                y = int(point.y * h)
                if 5 <= x < w-5 and 5 <= y < h-5:
                    roi = image[y-5:y+5, x-5:x+5]
                    if roi.size > 0:
                        avg_color = np.mean(roi, axis=(0, 1))
                        if np.all(avg_color > 20) and np.all(avg_color < 250):
                            color_samples.append(avg_color)
            
            if color_samples:
                avg_color_per_channel = np.mean(color_samples, axis=0)
                avg_color = (avg_color_per_channel[2], avg_color_per_channel[1], avg_color_per_channel[0])
                
                debug_data["skin_tone_analysis"] = {
                    "color_samples_count": len(color_samples),
                    "average_rgb": [round(c, 1) for c in avg_color],
                    "red_value": round(avg_color[0], 1),
                    "red_green_diff": round(avg_color[0] - avg_color[1], 1),
                    "red_blue_diff": round(avg_color[0] - avg_color[2], 1)
                }
            
            # Eye analysis debug info
            left_eye_outer = (int(landmarks[33].x * w), int(landmarks[33].y * h))
            left_eye_inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
            left_eye_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            left_eye_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            
            right_eye_outer = (int(landmarks[362].x * w), int(landmarks[362].y * h))
            right_eye_inner = (int(landmarks[263].x * w), int(landmarks[263].y * h))
            right_eye_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            right_eye_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))
            
            left_eye_width = np.linalg.norm(np.array(left_eye_outer) - np.array(left_eye_inner))
            left_eye_height = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
            right_eye_width = np.linalg.norm(np.array(right_eye_outer) - np.array(right_eye_inner))
            right_eye_height = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))
            
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            eye_ratio = avg_eye_height / avg_eye_width if avg_eye_width > 0 else 0
            
            debug_data["eye_analysis"] = {
                "left_eye_width": round(left_eye_width, 2),
                "left_eye_height": round(left_eye_height, 2),
                "right_eye_width": round(right_eye_width, 2),
                "right_eye_height": round(right_eye_height, 2),
                "avg_eye_width": round(avg_eye_width, 2),
                "avg_eye_height": round(avg_eye_height, 2),
                "eye_ratio": round(eye_ratio, 3)
            }
            
            # Lip analysis debug info
            upper_lip_center = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            upper_lip_left = (int(landmarks[78].x * w), int(landmarks[78].y * h))
            upper_lip_right = (int(landmarks[308].x * w), int(landmarks[308].y * h))
            upper_lip_top = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            
            lower_lip_center = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            lower_lip_left = (int(landmarks[84].x * w), int(landmarks[84].y * h))
            lower_lip_right = (int(landmarks[314].x * w), int(landmarks[314].y * h))
            lower_lip_bottom = (int(landmarks[17].x * w), int(landmarks[17].y * h))
            
            # Calculate lip dimensions
            lip_width = np.linalg.norm(np.array(upper_lip_left) - np.array(upper_lip_right))
            lip_height = np.linalg.norm(np.array(upper_lip_top) - np.array(lower_lip_bottom))
            
            # Additional measurements for better classification
            upper_lip_midpoint = (np.array(upper_lip_left) + np.array(upper_lip_right)) / 2
            lower_lip_midpoint = (np.array(lower_lip_left) + np.array(lower_lip_right)) / 2
            upper_lip_height = np.linalg.norm(np.array(upper_lip_center) - upper_lip_midpoint)
            lower_lip_height = np.linalg.norm(np.array(lower_lip_center) - lower_lip_midpoint)
            
            total_lip_height = upper_lip_height + lower_lip_height
            
            debug_data["lip_analysis"] = {
                "lip_width": round(lip_width, 2),
                "lip_height": round(lip_height, 2),
                "lip_ratio": round(lip_height / lip_width, 3) if lip_width > 0 else 0,
                "upper_lip_height": round(upper_lip_height, 2),
                "lower_lip_height": round(lower_lip_height, 2),
                "total_lip_height": round(total_lip_height, 2),
                "height_width_ratio": round(total_lip_height / lip_width, 3) if lip_width > 0 else 0
            }
            
            # Nose analysis debug info
            nose_bridge_top = (int(landmarks[168].x * w), int(landmarks[168].y * h))
            nose_bridge_mid = (int(landmarks[6].x * w), int(landmarks[6].y * h))
            nose_bridge_lower = (int(landmarks[197].x * w), int(landmarks[197].y * h))
            nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            nose_bottom = (int(landmarks[2].x * w), int(landmarks[2].y * h))
            
            nose_left = (int(landmarks[129].x * w), int(landmarks[129].y * h))
            nose_right = (int(landmarks[358].x * w), int(landmarks[358].y * h))
            nose_base_left = (int(landmarks[131].x * w), int(landmarks[131].y * h))
            nose_base_right = (int(landmarks[360].x * w), int(landmarks[360].y * h))
            nose_bridge_left = (int(landmarks[5].x * w), int(landmarks[5].y * h))
            nose_bridge_right = (int(landmarks[195].x * w), int(landmarks[195].y * h))
            nose_tip_left = (int(landmarks[125].x * w), int(landmarks[125].y * h))
            nose_tip_right = (int(landmarks[356].x * w), int(landmarks[356].y * h))
            
            nose_length = np.linalg.norm(np.array(nose_bridge_top) - np.array(nose_bottom))
            nose_width = np.linalg.norm(np.array(nose_left) - np.array(nose_right))
            nose_base_width = np.linalg.norm(np.array(nose_base_left) - np.array(nose_base_right))
            bridge_width = np.linalg.norm(np.array(nose_bridge_left) - np.array(nose_bridge_right))
            tip_width = np.linalg.norm(np.array(nose_tip_left) - np.array(nose_tip_right))
            
            if nose_length <= 0 or nose_width <= 0:
                debug_data["nose_analysis"] = {"error": "Invalid nose measurements"}
            else:
                # Calculate detailed ratios and characteristics
                nose_ratio = nose_length / nose_width
                base_ratio = nose_base_width / nose_width
                bridge_tip_ratio = bridge_width / tip_width if tip_width > 0 else 1
                
                # Bridge straightness analysis
                bridge_points = [
                    np.array(nose_bridge_top),
                    np.array(nose_bridge_mid),
                    np.array(nose_bridge_lower),
                    np.array(nose_tip)
                ]
                
                # Calculate bridge curvature
                bridge_curvature = 0
                for i in range(1, len(bridge_points) - 1):
                    v1 = bridge_points[i] - bridge_points[i-1]
                    v2 = bridge_points[i+1] - bridge_points[i]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)  # Prevent domain error
                        angle = np.arccos(cos_angle)
                        bridge_curvature += angle
                
                # Tip angle analysis (upturned/downturned)
                tip_angle = np.arctan2(nose_tip[1] - nose_bridge_lower[1], 
                                     nose_tip[0] - nose_bridge_lower[0])
                
                # Asymmetry detection
                left_side = np.linalg.norm(np.array(nose_bridge_left) - np.array(nose_tip_left))
                right_side = np.linalg.norm(np.array(nose_bridge_right) - np.array(nose_tip_right))
                asymmetry = abs(left_side - right_side) / max(left_side, right_side) if max(left_side, right_side) > 0 else 0
            
            debug_data["nose_analysis"] = {
                "nose_length": round(nose_length, 2),
                "nose_width": round(nose_width, 2),
                "nose_ratio": round(nose_ratio, 3),
                    "bridge_width": round(bridge_width, 2),
                    "tip_width": round(tip_width, 2),
                    "bridge_tip_ratio": round(bridge_tip_ratio, 3),
                    "base_ratio": round(base_ratio, 3),
                    "bridge_curvature": round(bridge_curvature, 3),
                    "tip_angle": round(tip_angle, 3),
                    "asymmetry": round(asymmetry, 3),
                    "measurements": {
                        "bridge_top": nose_bridge_top,
                        "bridge_mid": nose_bridge_mid,
                        "bridge_lower": nose_bridge_lower,
                        "nose_tip": nose_tip,
                        "nose_bottom": nose_bottom,
                        "nose_left": nose_left,
                        "nose_right": nose_right
                    }
                }
            
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error in nose shape analysis: {e}")
            return "Unknown"
        
        return debug_data

def analyze_skin_type(image):
    # Enhanced skin type analysis with detailed subtypes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple analysis methods for comprehensive detection
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Enhanced texture analysis
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    eroded = cv2.erode(gray, kernel, iterations=1)
    texture_variance = np.var(dilated - eroded)
    
    # Color analysis for skin type detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1])
    value = np.mean(hsv[:,:,2])
    
    # Comprehensive skin type classification with detailed subtypes
    if variance > 800 or edge_density > 0.1 or texture_variance > 200:
        if saturation > 50 and value > 150:
            return "Oily"  # High texture, shine, and saturation
        else:
            return "Combination"  # High texture but mixed characteristics
    elif variance < 150 or edge_density < 0.02:
        if value < 100:
            return "Dry"   # Low texture, dull appearance
        else:
            return "Sensitive"  # Low texture but normal brightness
    elif 200 < variance < 600 and 0.02 < edge_density < 0.08:
        if texture_variance < 100:
            return "Normal"  # Balanced characteristics
        else:
            return "Combination"  # Mixed characteristics
    else:
        # Additional logic for edge cases
        if saturation > 60:
            return "Oily"
        elif saturation < 30:
            return "Dry"
        elif texture_variance > 150:
            return "Combination"
        else:
            return "Normal"

def analyze_skin_tone(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Unknown", "Unknown"

        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # More comprehensive keypoints for skin tone analysis
            keypoints = [
                landmarks[10],   # Forehead center
                landmarks[234],  # Left cheek
                landmarks[454],  # Right cheek
                landmarks[152],  # Chin
                landmarks[132],  # Left temple
                landmarks[361],  # Right temple
                landmarks[199],  # Left jaw
                landmarks[419]   # Right jaw
            ]

            color_samples = []
            for point in keypoints:
                x = int(point.x * w)
                y = int(point.y * h)
                
                # Ensure coordinates are within image bounds
                if 5 <= x < w-5 and 5 <= y < h-5:
                    roi = image[y-5:y+5, x-5:x+5]
                    if roi.size > 0:
                        avg_color = np.mean(roi, axis=(0, 1))
                        # Filter out extreme values (likely not skin)
                        if np.all(avg_color > 20) and np.all(avg_color < 250):
                            color_samples.append(avg_color)

            if len(color_samples) < 3:  # Need at least 3 valid samples
                return "Unknown", "Unknown"
                
            avg_color_per_channel = np.mean(color_samples, axis=0)
            
            # BGR to RGB conversion
            avg_color = (avg_color_per_channel[2], avg_color_per_channel[1], avg_color_per_channel[0])

            # Enhanced skin tone classification with corrected thresholds
            red_value = avg_color[0]
            green_value = avg_color[1]
            blue_value = avg_color[2]
            
            # Calculate luminance for better skin tone detection
            luminance = 0.299 * red_value + 0.587 * green_value + 0.114 * blue_value
            
            # Enhanced skin tone classification with corrected thresholds
            if luminance > 180:  # Adjusted threshold
                skin_tone = "Very Fair"
            elif luminance > 150:  # Adjusted threshold
                skin_tone = "Fair"
            elif luminance > 120:  # Adjusted threshold for medium
                # Check for olive undertone
                if abs(red_value - green_value) < 15 and green_value > blue_value:
                    skin_tone = "Medium (Olive)"
                else:
                    skin_tone = "Medium"
            elif luminance > 90:  # Adjusted threshold
                skin_tone = "Tan"
            elif luminance > 70:  # Adjusted threshold
                skin_tone = "Deep Tan"
            else:
                skin_tone = "Deep"

            # Enhanced undertone classification with corrected analysis
            red_green_diff = red_value - green_value
            red_blue_diff = red_value - blue_value
            green_blue_diff = green_value - blue_value
            
            # Calculate color temperature
            color_temp = (red_value + green_value) / 2 - blue_value
            
            # Enhanced undertone classification with corrected thresholds
            if red_green_diff > 20 and red_blue_diff > 20 and color_temp > 15:  # Adjusted thresholds
                undertone = "Warm"  # Yellow, peach, or golden
            elif red_blue_diff < -15 and green_blue_diff < -10:  # Adjusted thresholds
                undertone = "Cool"  # Pink, red, or blue
            elif abs(red_green_diff) < 10 and abs(red_blue_diff) < 10:  # Adjusted thresholds
                undertone = "Neutral"  # Balanced mix of both
            else:
                # Additional logic for edge cases
                if color_temp > 10:  # Adjusted threshold
                    undertone = "Warm"
                elif color_temp < -10:  # Adjusted threshold
                    undertone = "Cool"
                else:
                    undertone = "Neutral"
                
            return skin_tone, undertone
            
        except (IndexError, ValueError) as e:
            print(f"Error in skin tone analysis: {e}")
            return "Unknown", "Unknown"

def analyze_face_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "No face detected"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        try:
            # Enhanced landmark points for comprehensive face shape analysis
            forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
            chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            jaw_left = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            jaw_right = (int(landmarks[454].x * w), int(landmarks[454].y * h))
            cheek_left = (int(landmarks[132].x * w), int(landmarks[132].y * h))
            cheek_right = (int(landmarks[361].x * w), int(landmarks[361].y * h))
            
            # Additional landmarks for better shape detection
            temple_left = (int(landmarks[447].x * w), int(landmarks[447].y * h))
            temple_right = (int(landmarks[227].x * w), int(landmarks[227].y * h))
            jaw_angle_left = (int(landmarks[199].x * w), int(landmarks[199].y * h))
            jaw_angle_right = (int(landmarks[419].x * w), int(landmarks[419].y * h))
            
            # Calculate comprehensive measurements
            face_width = np.linalg.norm(np.array(jaw_left) - np.array(jaw_right))
            face_height = chin[1] - forehead[1]
            cheek_width = np.linalg.norm(np.array(cheek_left) - np.array(cheek_right))
            temple_width = np.linalg.norm(np.array(temple_left) - np.array(temple_right))
            jaw_width = np.linalg.norm(np.array(jaw_angle_left) - np.array(jaw_angle_right))
            
            if face_width <= 0 or face_height <= 0 or cheek_width <= 0:
                return "Could not determine shape - invalid measurements"
            
            # Enhanced ratios for better classification
            height_width_ratio = face_height / face_width
            cheek_jaw_ratio = cheek_width / face_width
            temple_jaw_ratio = temple_width / jaw_width
            cheek_temple_ratio = cheek_width / temple_width
            
            # Comprehensive face shape classification with corrected thresholds
            if 1.2 <= height_width_ratio <= 1.6:  # Adjusted range for oval
                if cheek_jaw_ratio > 1.0 and temple_jaw_ratio > 1.0:
                    return "Oval"  # Balanced proportions
                elif height_width_ratio > 1.5:
                    return "Rectangle (Oblong)"  # Long and narrow
                else:
                    return "Oval"
            elif 0.9 <= height_width_ratio <= 1.2:
                if cheek_jaw_ratio > 1.15 and temple_jaw_ratio > 1.1:
                    return "Round"  # Wide and circular
                elif abs(cheek_jaw_ratio - 1.0) < 0.1 and abs(temple_jaw_ratio - 1.0) < 0.1:
                    return "Square"  # Equal width at all levels
                else:
                    return "Round"
            elif height_width_ratio < 0.9:
                return "Square"  # Wide and angular
            elif height_width_ratio > 1.6:
                return "Rectangle (Oblong)"  # Very long and narrow
            elif temple_jaw_ratio > 1.2 and cheek_temple_ratio > 1.1:
                return "Heart"  # Wide forehead, narrow jaw
            elif cheek_jaw_ratio > 1.2 and temple_jaw_ratio < 0.9:
                return "Diamond"  # Wide cheekbones, narrow forehead and jaw
            elif temple_jaw_ratio < 0.8 and cheek_jaw_ratio < 0.9:
                return "Triangle (Pear)"  # Narrow forehead, wide jaw
            else:
                # Additional logic for edge cases
                if cheek_jaw_ratio > 1.1:
                    return "Heart"
                elif temple_jaw_ratio > 1.1:
                    return "Diamond"
                else:
                    return "Oval"  # Default fallback
                
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error in face shape analysis: {e}")
            return "Could not determine shape - analysis error"

def analyze_eye_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        try:
            # Enhanced eye shape analysis using comprehensive landmarks
            # Left eye landmarks
            left_eye_outer = (int(landmarks[33].x * w), int(landmarks[33].y * h))
            left_eye_inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
            left_eye_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            left_eye_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            left_eye_upper_lid = (int(landmarks[158].x * w), int(landmarks[158].y * h))
            left_eye_lower_lid = (int(landmarks[144].x * w), int(landmarks[144].y * h))
            
            # Right eye landmarks
            right_eye_outer = (int(landmarks[362].x * w), int(landmarks[362].y * h))
            right_eye_inner = (int(landmarks[263].x * w), int(landmarks[263].y * h))
            right_eye_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            right_eye_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))
            right_eye_upper_lid = (int(landmarks[385].x * w), int(landmarks[385].y * h))
            right_eye_lower_lid = (int(landmarks[373].x * w), int(landmarks[373].y * h))
            
            # Calculate comprehensive eye measurements
            left_eye_width = np.linalg.norm(np.array(left_eye_outer) - np.array(left_eye_inner))
            left_eye_height = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
            right_eye_width = np.linalg.norm(np.array(right_eye_outer) - np.array(right_eye_inner))
            right_eye_height = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))
            
            # Average eye dimensions
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            if avg_eye_width <= 0 or avg_eye_height <= 0:
                return "Unknown"
            
            # Calculate eye ratios and characteristics
            eye_ratio = avg_eye_height / avg_eye_width
            
            # Eye spacing analysis
            eye_spacing = np.linalg.norm(np.array(left_eye_inner) - np.array(right_eye_inner))
            eye_spacing_ratio = eye_spacing / avg_eye_width
            
            # Eye depth analysis (distance from eye to brow)
            left_brow = (int(landmarks[70].x * w), int(landmarks[70].y * h))
            right_brow = (int(landmarks[300].x * w), int(landmarks[300].y * h))
            left_eye_depth = abs(left_eye_top[1] - left_brow[1])
            right_eye_depth = abs(right_eye_top[1] - right_brow[1])
            avg_eye_depth = (left_eye_depth + right_eye_depth) / 2
            
            # Eye angle analysis (upturned/downturned)
            left_eye_angle = np.arctan2(left_eye_outer[1] - left_eye_inner[1], 
                                      left_eye_outer[0] - left_eye_inner[0])
            right_eye_angle = np.arctan2(right_eye_inner[1] - right_eye_outer[1], 
                                       right_eye_inner[0] - right_eye_outer[0])
            avg_eye_angle = (left_eye_angle + right_eye_angle) / 2
            
            # Comprehensive eye shape classification with corrected thresholds
            # First, determine basic shape based on corrected ratios
            if eye_ratio > 0.5:  # Increased threshold for round eyes
                base_shape = "Round"
            elif eye_ratio < 0.35:  # Adjusted threshold for almond eyes
                base_shape = "Almond"
            else:
                base_shape = "Almond"  # Default to almond for most cases
            
            # Determine additional characteristics
            characteristics = []
            
            # Eye spacing
            if eye_spacing_ratio < 1.5:
                characteristics.append("Close-set")
            elif eye_spacing_ratio > 2.5:
                characteristics.append("Wide-set")
            
            # Eye depth
            if avg_eye_depth > avg_eye_height * 0.8:
                characteristics.append("Deep-set")
            elif avg_eye_depth < avg_eye_height * 0.3:
                characteristics.append("Protruding")
            
            # Eye angle (upturned/downturned)
            if avg_eye_angle > 0.3:  # Positive angle indicates upturned
                characteristics.append("Upturned")
            elif avg_eye_angle < -0.3:  # Negative angle indicates downturned
                characteristics.append("Downturned")
            
            # Hooded eyes detection (upper lid covers more of the eye)
            left_hood_ratio = left_eye_upper_lid[1] / left_eye_height
            right_hood_ratio = right_eye_upper_lid[1] / right_eye_height
            avg_hood_ratio = (left_hood_ratio + right_hood_ratio) / 2
            
            if avg_hood_ratio > 0.6:
                characteristics.append("Hooded")
            
            # Monolid detection (minimal crease)
            if avg_eye_height < avg_eye_width * 0.2:
                characteristics.append("Monolid")
            
            # Combine characteristics
            if characteristics:
                return f"{base_shape} ({', '.join(characteristics)})"
            else:
                return base_shape
                
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error in eye shape analysis: {e}")
            return "Unknown"

def analyze_lip_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        try:
            # Enhanced lip shape analysis using comprehensive MediaPipe landmarks
            upper_lip_center = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            upper_lip_left = (int(landmarks[78].x * w), int(landmarks[78].y * h))
            upper_lip_right = (int(landmarks[308].x * w), int(landmarks[308].y * h))
            upper_lip_top = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            upper_lip_cupid = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            lower_lip_center = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            lower_lip_left = (int(landmarks[84].x * w), int(landmarks[84].y * h))
            lower_lip_right = (int(landmarks[314].x * w), int(landmarks[314].y * h))
            lower_lip_bottom = (int(landmarks[17].x * w), int(landmarks[17].y * h))
            upper_lip_mid_left = (int(landmarks[81].x * w), int(landmarks[81].y * h))
            upper_lip_mid_right = (int(landmarks[311].x * w), int(landmarks[311].y * h))
            lower_lip_mid_left = (int(landmarks[87].x * w), int(landmarks[87].y * h))
            lower_lip_mid_right = (int(landmarks[317].x * w), int(landmarks[317].y * h))

            lip_width = np.linalg.norm(np.array(upper_lip_left) - np.array(upper_lip_right))
            lip_height = np.linalg.norm(np.array(upper_lip_top) - np.array(lower_lip_bottom))
            upper_lip_midpoint = (np.array(upper_lip_left) + np.array(upper_lip_right)) / 2
            lower_lip_midpoint = (np.array(lower_lip_left) + np.array(lower_lip_right)) / 2
            upper_lip_height = np.linalg.norm(np.array(upper_lip_center) - upper_lip_midpoint)
            lower_lip_height = np.linalg.norm(np.array(lower_lip_center) - lower_lip_midpoint)
            
            if lip_width <= 0 or lip_height <= 0:
                return "Unknown"
            
            lip_ratio = lip_height / lip_width
            total_lip_height = upper_lip_height + lower_lip_height
            upper_lower_ratio = upper_lip_height / lower_lip_height if lower_lip_height > 0 else 1
            cupid_height = abs(upper_lip_cupid[1] - upper_lip_center[1])
            cupid_ratio = cupid_height / upper_lip_height if upper_lip_height > 0 else 0
            fullness_ratio = total_lip_height / lip_width

            # Prioritize specific subtypes
            subtypes = []
            if upper_lower_ratio > 1.2:
                subtypes.append("Top-heavy")
            elif upper_lower_ratio < 0.6:
                subtypes.append("Bottom-heavy")
            if cupid_ratio > 0.4 and upper_lower_ratio > 1.1:
                subtypes.append("Heart-shaped")
            if lip_ratio > 0.5:
                subtypes.append("Round")
            if lip_width > w * 0.4:  # Compare to face width
                subtypes.append("Wide")
            if lip_ratio < 0.2:
                subtypes.append("Thin")
            if fullness_ratio > 0.6:
                subtypes.append("Full")

            # If no specific subtype, fallback to Full/Thin/Medium
            if not subtypes:
                if fullness_ratio > 0.45 or lip_ratio > 0.4:
                    return "Full"
                elif fullness_ratio < 0.2 or lip_ratio < 0.25:
                    return "Thin"
                else:
                    return "Medium"
            else:
                return ", ".join(subtypes)
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error in lip shape analysis: {e}")
            return "Unknown"

def analyze_nose_contouring_needs(nose_shape, face_shape):
    """Determine if nose contouring is needed and provide specific guidance"""
    contouring_analysis = {
        "wide": {
            "needs_contouring": True,
            "contouring_areas": [
                "Along the sides of the nose bridge",
                "Under the nose tip",
                "Around the nostrils"
            ],
            "technique": "Apply contour along the sides of the nose bridge to make it appear narrower",
            "highlight_areas": ["Down the center of the nose bridge", "Tip of the nose"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour along the sides of the nose bridge",
                "Blend thoroughly for a natural look",
                "Apply highlight down the center of the nose",
                "Highlight the tip of the nose",
                "Blend everything seamlessly"
            ]
        },
        "long": {
            "needs_contouring": True,
            "contouring_areas": [
                "Under the nose tip",
                "Across the bridge to create breaks"
            ],
            "technique": "Apply contour under the nose tip to shorten the appearance",
            "highlight_areas": ["Center of the nose bridge"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour under the nose tip",
                "Apply contour across the bridge to create visual breaks",
                "Highlight the center of the nose bridge",
                "Blend thoroughly for a natural look"
            ]
        },
        "medium": {
            "needs_contouring": False,
            "contouring_areas": [],
            "technique": "Your nose has balanced proportions. Light highlighting can enhance its natural shape.",
            "highlight_areas": ["Center of the nose bridge", "Tip of the nose"],
            "products": [
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"},
                {"name": "Lakme Absolute Illuminating Highlighter", "type": "Highlight", "price": "PKR 680", "brand": "Lakme"}
            ],
            "steps": [
                "Apply light highlight down the center of the nose",
                "Highlight the tip of the nose",
                "Blend for a subtle glow"
            ]
        }
    }
    
    return contouring_analysis.get(nose_shape.lower(), {
        "needs_contouring": False,
        "contouring_areas": [],
        "technique": "Your nose has natural proportions. Light highlighting can enhance its shape.",
        "highlight_areas": ["Center of the nose bridge"],
        "steps": ["Apply light highlight down the center of the nose", "Blend for a subtle glow"],
        "products": []
    })

def analyze_face_contouring_needs(face_shape):
    """Determine if face contouring is needed and provide specific guidance"""
    face_contouring_analysis = {
        "round": {
            "needs_contouring": True,
            "contouring_areas": [
                "Along the jawline",
                "Under the cheekbones",
                "Along the hairline",
                "Under the chin"
            ],
            "technique": "Create definition and angles to balance the round shape",
            "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour along the jawline",
                "Contour under the cheekbones",
                "Apply contour along the hairline",
                "Contour under the chin",
                "Highlight the center of forehead, under eyes, and chin",
                "Blend thoroughly for a natural look"
            ]
        },
        "square": {
            "needs_contouring": True,
            "contouring_areas": [
                "Along the jawline to soften angles",
                "Under the cheekbones",
                "Along the hairline"
            ],
            "technique": "Soften the angular features and create more curves",
            "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour along the jawline to soften angles",
                "Contour under the cheekbones",
                "Apply contour along the hairline",
                "Highlight the center of forehead, under eyes, and chin",
                "Blend thoroughly for a natural look"
            ]
        },
        "heart": {
            "needs_contouring": True,
            "contouring_areas": [
                "Along the jawline",
                "Under the cheekbones",
                "Along the hairline at the temples"
            ],
            "technique": "Balance the wider forehead and narrower jaw",
            "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour along the jawline",
                "Contour under the cheekbones",
                "Apply contour along the hairline at the temples",
                "Highlight the center of forehead, under eyes, and chin",
                "Blend thoroughly for a natural look"
            ]
        },
        "oval": {
            "needs_contouring": False,
            "contouring_areas": [],
            "technique": "Your face has balanced proportions. Light highlighting can enhance your natural features.",
            "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin", "Top of cheekbones"],
            "products": [
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"},
                {"name": "Lakme Absolute Illuminating Highlighter", "type": "Highlight", "price": "PKR 680", "brand": "Lakme"}
            ],
            "steps": [
                "Apply highlight to the center of forehead",
                "Highlight under the eyes",
                "Apply highlight to the center of chin",
                "Highlight the top of cheekbones",
                "Blend for a natural glow"
            ]
        },
        "diamond": {
            "needs_contouring": True,
            "contouring_areas": [
                "Along the hairline",
                "Under the cheekbones",
                "Along the jawline"
            ],
            "technique": "Balance the wider cheekbones and narrower forehead/jaw",
            "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
            "products": [
                {"name": "Lakme Absolute Precision Contour Stick", "type": "Contour", "price": "PKR 750", "brand": "Lakme"},
                {"name": "Maybelline Fit Me Concealer", "type": "Highlight", "price": "PKR 580", "brand": "Maybelline"}
            ],
            "steps": [
                "Apply contour along the hairline",
                "Contour under the cheekbones",
                "Apply contour along the jawline",
                "Highlight the center of forehead, under eyes, and chin",
                "Blend thoroughly for a natural look"
            ]
        }
    }
    
    return face_contouring_analysis.get(face_shape.lower(), {
        "needs_contouring": False,
        "contouring_areas": [],
        "technique": "Your face has natural proportions. Light highlighting can enhance your features.",
        "highlight_areas": ["Center of forehead", "Under eyes", "Center of chin"],
        "steps": ["Apply light highlight to key areas", "Blend for a subtle glow"],
        "products": []
    })

def get_comprehensive_recommendations(skin_type, skin_tone, undertone, face_shape, eye_shape, lip_shape, nose_shape):
    # Get skincare recommendations
    skin_type_rec = products['skincare'].get(skin_type.lower(), [])
    
    # Get foundation recommendation
    tone_key = f"{skin_tone.lower()}_{undertone.lower()}"
    foundation_rec = products['makeup']['foundation'].get(tone_key, {"name": "N/A", "link": "#", "price": "N/A"})
    
    # Get lipstick recommendations
    lipstick_rec = products['makeup']['lipstick'].get(tone_key, [])
    
    # Get skin tone-specific lip shades
    lip_shades_rec = products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {})
    
    # Get skin tone-specific blush recommendations
    blush_by_tone_rec = products['makeup']['blush_by_skin_tone'].get(tone_key, {})
    
    # Get blush recommendations (face shape based)
    blush_rec = products['makeup']['blush'].get(face_shape.lower(), {})
    
    # Get contouring recommendation
    contour_style = "natural"  # Default to natural
    if face_shape.lower() in ["square", "round"]:
        contour_style = "sculpted"
    elif face_shape.lower() in ["heart", "diamond"]:
        contour_style = "soft"
    
    contour_rec = products['makeup']['contouring'].get(contour_style, {})
    
    # Get eyeliner recommendations
    eyeliner_rec = products['makeup']['eyeliner'].get(eye_shape.lower(), {})
    
    # Get nose contouring recommendations
    nose_contour_rec = products['makeup']['nose_contouring'].get(nose_shape.lower(), {})
    
    # Combine all makeup recommendations into a flat array
    makeup_recommendations = []
    
    # Add foundation
    if foundation_rec and foundation_rec.get("name") != "N/A":
        makeup_recommendations.append(foundation_rec)
    
    # Add lipstick recommendations
    if lipstick_rec:
        makeup_recommendations.extend(lipstick_rec)
    
    # Add lip shades - extract from recommended_shades
    if lip_shades_rec and isinstance(lip_shades_rec, dict):
        recommended_shades = lip_shades_rec.get('recommended_shades', [])
        if recommended_shades:
            makeup_recommendations.extend(recommended_shades)
    
    # Add blush recommendations - extract from recommended_shades
    if blush_by_tone_rec and isinstance(blush_by_tone_rec, dict):
        recommended_shades = blush_by_tone_rec.get('recommended_shades', [])
        if recommended_shades:
            makeup_recommendations.extend(recommended_shades)
    
    # Add blush recommendations from face shape
    if blush_rec:
        if isinstance(blush_rec, list):
            makeup_recommendations.extend(blush_rec)
        elif isinstance(blush_rec, dict):
            # Extract products from blush dict
            products_list = blush_rec.get('products', [])
            if products_list:
                makeup_recommendations.extend(products_list)
    
    # Add contouring
    if contour_rec:
        if isinstance(contour_rec, list):
            makeup_recommendations.extend(contour_rec)
        elif isinstance(contour_rec, dict):
            # Extract products from contour dict
            products_list = contour_rec.get('products', [])
            if products_list:
                makeup_recommendations.extend(products_list)
    
    # Add eyeliner
    if eyeliner_rec:
        if isinstance(eyeliner_rec, list):
            makeup_recommendations.extend(eyeliner_rec)
        elif isinstance(eyeliner_rec, dict):
            # Extract products from eyeliner dict
            products_list = eyeliner_rec.get('products', [])
            if products_list:
                makeup_recommendations.extend(products_list)
    
    # Add nose contouring
    if nose_contour_rec:
        if isinstance(nose_contour_rec, list):
            makeup_recommendations.extend(nose_contour_rec)
        elif isinstance(nose_contour_rec, dict):
            # Extract products from nose contour dict
            products_list = nose_contour_rec.get('products', [])
            if products_list:
                makeup_recommendations.extend(products_list)
    
    # Create personalized tips as simple strings
    tips = []
    
    # Face shape tips
    if face_shape.lower() != "unknown":
        tips.append(f"Perfect your {face_shape.lower()} face shape with contouring techniques that define your features.")
        tips.append(f"Apply blush to the apples of your cheeks to enhance your {face_shape.lower()} face shape.")
        tips.append(f"Choose hairstyles that complement your {face_shape.lower()} face shape for the best look.")
    
    # Skin tone tips
    if skin_tone.lower() != "unknown":
        tips.append(f"Your {skin_tone.lower()} skin tone works beautifully with warm and cool color palettes.")
        tips.append(f"Choose foundation that matches your {skin_tone.lower()} undertone for a natural finish.")
        tips.append(f"Select lip colors and blush shades that enhance your {skin_tone.lower()} skin tone.")
    
    # Eye shape tips
    if eye_shape.lower() != "unknown":
        tips.append(f"Your {eye_shape.lower()} eyes can be enhanced with specific eyeliner and shadow techniques.")
        tips.append(f"Apply eyeliner techniques perfect for {eye_shape.lower()} eyes to make them pop.")
        tips.append(f"Choose eyeshadow colors that complement your {eye_shape.lower()} eye shape.")
    
    # Lip shape tips
    if lip_shape.lower() != "unknown":
        tips.append(f"Your {lip_shape.lower()} lips can be beautifully defined with the right lip products.")
        tips.append(f"Use lip liner to enhance your {lip_shape.lower()} lip shape for perfect definition.")
        tips.append(f"Choose lipstick formulas that work best for your {lip_shape.lower()} lips.")
    
    # Skincare tips
    if skin_type.lower() != "unknown":
        tips.append(f"Your {skin_type.lower()} skin requires specific care and attention for optimal health.")
        tips.append(f"Use products specifically formulated for {skin_type.lower()} skin for best results.")
        tips.append(f"Follow a daily skincare routine tailored to your {skin_type.lower()} skin type.")
    
    return {
        "skincare": skin_type_rec,
        "makeup": makeup_recommendations,
        "tips": tips
    }

def analyze_lip_colors_for_skin_tone(skin_tone, undertone):
    """Analyze which lip colors suit specific skin tones"""
    tone_key = f"{skin_tone.lower()}_{undertone.lower()}"
    
    lip_recommendations = {
        "fair_warm": {
            "best_colors": ["Peachy Pink", "Coral", "Warm Nude", "Mauve", "Warm Brown"],
            "avoid_colors": ["Cool Blue-based Reds", "Purple", "Cool Pink"],
            "technique": "Apply with a lip liner for definition. Warm undertones look best with golden and peachy tones.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "fair_cool": {
            "best_colors": ["Cool Pink", "Berry", "Blue-based Red", "Cool Coral", "Plum"],
            "avoid_colors": ["Orange", "Warm Brown", "Terracotta"],
            "technique": "Cool undertones are enhanced by blue-based colors. Apply with precision for a polished look.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "medium_warm": {
            "best_colors": ["Terracotta", "Warm Red", "Golden Nude", "Warm Spice", "Coral"],
            "avoid_colors": ["Cool Pink", "Purple", "Blue-based Red"],
            "technique": "Medium warm skin looks stunning with rich, warm tones. Blend well for a natural finish.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "medium_cool": {
            "best_colors": ["Berry", "Mauve", "Cool Red", "Plum", "Cool Rose"],
            "avoid_colors": ["Orange", "Warm Brown", "Terracotta"],
            "technique": "Cool undertones in medium skin look elegant with berry and mauve tones.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "deep_warm": {
            "best_colors": ["Rich Burgundy", "Warm Brown", "Golden Nude", "Terracotta", "Warm Mahogany"],
            "avoid_colors": ["Light Pink", "Cool Pink", "Purple"],
            "technique": "Deep warm skin looks regal with rich, warm tones. Use bold colors for maximum impact.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "deep_cool": {
            "best_colors": ["Deep Plum", "Cool Burgundy", "Rich Berry", "Deep Wine", "Cool Mahogany"],
            "avoid_colors": ["Light Pink", "Orange", "Warm Brown"],
            "technique": "Deep cool skin looks sophisticated with deep, cool tones. Apply with confidence.",
            "products": products['makeup']['lip_shades_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        }
    }
    
    return lip_recommendations.get(tone_key, {
        "best_colors": ["Universal Nude", "Classic Red", "Soft Pink"],
        "avoid_colors": ["Extreme colors"],
        "technique": "Choose colors that complement your natural undertone.",
        "products": []
    })

def analyze_blush_for_skin_tone(skin_tone, undertone):
    """Analyze which blush shades complement different skin tones"""
    tone_key = f"{skin_tone.lower()}_{undertone.lower()}"
    
    blush_recommendations = {
        "fair_warm": {
            "best_shades": ["Peach", "Coral", "Warm Pink", "Golden Peach"],
            "avoid_shades": ["Deep Berry", "Purple", "Cool Pink"],
            "application": "Apply to the apples of cheeks and blend upward toward temples. Use a light hand for natural look.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "fair_cool": {
            "best_shades": ["Soft Pink", "Berry", "Cool Rose", "Light Mauve"],
            "avoid_shades": ["Orange", "Warm Peach", "Terracotta"],
            "application": "Apply to the apples of cheeks and blend upward. Cool undertones look best with soft, cool tones.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "medium_warm": {
            "best_shades": ["Terracotta", "Warm Rose", "Coral", "Golden Peach"],
            "avoid_shades": ["Cool Pink", "Purple", "Cool Berry"],
            "application": "Apply to the apples of cheeks and blend upward. Medium warm skin can handle richer tones.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "medium_cool": {
            "best_shades": ["Mauve", "Cool Rose", "Berry", "Plum"],
            "avoid_shades": ["Orange", "Warm Peach", "Terracotta"],
            "application": "Apply to the apples of cheeks and blend upward. Cool undertones look elegant with mauve tones.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "deep_warm": {
            "best_shades": ["Rich Terracotta", "Warm Berry", "Deep Coral", "Mahogany"],
            "avoid_shades": ["Light Pink", "Cool Pink", "Purple"],
            "application": "Apply to the apples of cheeks and blend upward. Deep warm skin looks stunning with rich tones.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        },
        "deep_cool": {
            "best_shades": ["Deep Plum", "Cool Berry", "Deep Mauve", "Cool Burgundy"],
            "avoid_shades": ["Light Pink", "Orange", "Warm Peach"],
            "application": "Apply to the apples of cheeks and blend upward. Deep cool skin looks sophisticated with deep tones.",
            "products": products['makeup']['blush_by_skin_tone'].get(tone_key, {}).get('recommended_shades', [])
        }
    }
    
    return blush_recommendations.get(tone_key, {
        "best_shades": ["Universal Rose", "Soft Pink", "Natural Peach"],
        "avoid_shades": ["Extreme colors"],
        "application": "Apply to the apples of cheeks and blend upward for a natural flush.",
        "products": []
    })

def analyze_eye_makeup_for_eye_shape(eye_shape):
    """Recommend eye makeup styles for specific eye types"""
    eye_recommendations = {
        "round": {
            "eyeliner_style": "Winged liner to elongate the eye",
            "eyeshadow_technique": "Apply darker shades on the outer corners and blend upward",
            "mascara_technique": "Focus on outer lashes to create a cat-eye effect",
            "avoid": "Heavy eyeliner on the lower lash line",
            "products": [
                {"name": "Lakme Eyeconic Kajal - Black", "type": "Eyeliner", "price": "PKR 450", "brand": "Lakme"},
                {"name": "Maybelline Colossal Kajal - Black", "type": "Eyeliner", "price": "PKR 380", "brand": "Maybelline"},
                {"name": "L'Oreal Paris Super Liner - Black", "type": "Eyeliner", "price": "PKR 520", "brand": "L'Oreal"}
            ],
            "technique_steps": [
                "Apply a light base shadow all over the lid",
                "Use a medium shade in the crease to add depth",
                "Apply a darker shade on the outer corner and blend upward",
                "Create a winged liner starting from the outer corner",
                "Apply mascara focusing on the outer lashes"
            ]
        },
        "almond": {
            "eyeliner_style": "Classic liner along the upper lash line",
            "eyeshadow_technique": "Apply shadow in a V-shape on the outer corner",
            "mascara_technique": "Apply evenly to all lashes for natural enhancement",
            "avoid": "Overly dramatic looks that overwhelm the natural shape",
            "products": [
                {"name": "Lakme Absolute Precision Eyeliner", "type": "Eyeliner", "price": "PKR 580", "brand": "Lakme"},
                {"name": "Maybelline Hyper Precise Eyeliner", "type": "Eyeliner", "price": "PKR 450", "brand": "Maybelline"},
                {"name": "L'Oreal Paris Infallible Eyeliner", "type": "Eyeliner", "price": "PKR 620", "brand": "L'Oreal"}
            ],
            "technique_steps": [
                "Apply a neutral base shadow",
                "Use a medium shade in the crease",
                "Apply a darker shade in a V-shape on the outer corner",
                "Line the upper lash line with a classic liner",
                "Apply mascara evenly to all lashes"
            ]
        },
        "hooded": {
            "eyeliner_style": "Thin liner on the upper lash line, avoid thick lines",
            "eyeshadow_technique": "Apply shadow above the crease to make it visible",
            "mascara_technique": "Curl lashes and apply waterproof mascara",
            "avoid": "Thick eyeliner that covers the lid completely",
            "products": [
                {"name": "Lakme Absolute Precision Eyeliner", "type": "Eyeliner", "price": "PKR 580", "brand": "Lakme"},
                {"name": "Maybelline Hyper Precise Eyeliner", "type": "Eyeliner", "price": "PKR 450", "brand": "Maybelline"},
                {"name": "L'Oreal Paris Voluminous Mascara", "type": "Mascara", "price": "PKR 680", "brand": "L'Oreal"}
            ],
            "technique_steps": [
                "Apply a light base shadow above the crease",
                "Use a medium shade in the crease and blend upward",
                "Apply a darker shade on the outer corner",
                "Line the upper lash line with a thin line",
                "Curl lashes and apply waterproof mascara"
            ]
        }
    }
    
    return eye_recommendations.get(eye_shape.lower(), {
        "eyeliner_style": "Classic liner along the upper lash line",
        "eyeshadow_technique": "Apply shadow in the crease and outer corner",
        "mascara_technique": "Apply evenly to all lashes",
        "avoid": "Overly dramatic looks",
        "technique_steps": [
            "Apply a neutral base shadow",
            "Use a medium shade in the crease",
            "Apply a darker shade on the outer corner",
            "Line the upper lash line",
            "Apply mascara to all lashes"
        ],
        "products": []
    })

def analyze_nose_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        try:
            # Key nose landmarks
            nose_bridge_top = (int(landmarks[168].x * w), int(landmarks[168].y * h))
            nose_bridge_mid = (int(landmarks[6].x * w), int(landmarks[6].y * h))
            nose_bridge_lower = (int(landmarks[197].x * w), int(landmarks[197].y * h))
            nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            nose_bottom = (int(landmarks[2].x * w), int(landmarks[2].y * h))
            nose_left = (int(landmarks[129].x * w), int(landmarks[129].y * h))
            nose_right = (int(landmarks[358].x * w), int(landmarks[358].y * h))
            nose_bridge_left = (int(landmarks[5].x * w), int(landmarks[5].y * h))
            nose_bridge_right = (int(landmarks[195].x * w), int(landmarks[195].y * h))
            nose_tip_left = (int(landmarks[125].x * w), int(landmarks[125].y * h))
            nose_tip_right = (int(landmarks[356].x * w), int(landmarks[356].y * h))

            # Calculate measurements
            nose_length = np.linalg.norm(np.array(nose_bridge_top) - np.array(nose_bottom))
            nose_width = np.linalg.norm(np.array(nose_left) - np.array(nose_right))
            bridge_width = np.linalg.norm(np.array(nose_bridge_left) - np.array(nose_bridge_right))
            tip_width = np.linalg.norm(np.array(nose_tip_left) - np.array(nose_tip_right))

            if nose_length <= 0 or nose_width <= 0:
                return "Unknown"

            # Calculate ratios
            nose_ratio = nose_length / nose_width
            bridge_tip_ratio = bridge_width / tip_width if tip_width > 0 else 1

            # Bridge curvature analysis
            bridge_points = [
                np.array(nose_bridge_top),
                np.array(nose_bridge_mid),
                np.array(nose_bridge_lower),
                np.array(nose_tip)
            ]
            
            bridge_curvature = 0
            for i in range(1, len(bridge_points) - 1):
                v1 = bridge_points[i] - bridge_points[i-1]
                v2 = bridge_points[i+1] - bridge_points[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)  # Prevent domain error
                    angle = np.arccos(cos_angle)
                    bridge_curvature += angle

            # Tip angle analysis
            tip_angle = np.arctan2(nose_tip[1] - nose_bridge_lower[1], 
                                 nose_tip[0] - nose_bridge_lower[0])

            # Primary classification based on bridge shape
            if bridge_curvature > 0.5:  # High curvature
                primary_shape = "Roman"
            elif bridge_curvature < 0.2:  # Low curvature
                primary_shape = "Greek"
            else:
                primary_shape = "Straight"

            # Secondary characteristics
            characteristics = []
            
            # Tip characteristics
            if tip_angle > 0.2:
                characteristics.append("Upturned")
            elif tip_angle < -0.2:
                characteristics.append("Downturned")
            
            # Width characteristics
            if bridge_tip_ratio < 0.6:
                characteristics.append("Button")
            elif bridge_width < nose_width * 0.25:
                characteristics.append("Narrow")
            elif bridge_width > nose_width * 0.5:
                characteristics.append("Wide")
            
            # Length characteristics
            if nose_ratio > 4.5:
                characteristics.append("Long")
            elif nose_ratio < 2.8:
                characteristics.append("Short")
            
            # Special types
            if nose_ratio < 3.0 and tip_angle > 0.15:
                characteristics.append("Snub")

            # Combine primary shape with characteristics
            if characteristics:
                return f"{primary_shape} ({', '.join(characteristics)})"
            else:
                return primary_shape

        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error in nose shape analysis: {e}")
            return "Unknown"

if __name__ == '__main__':
    app.run(debug=True) 