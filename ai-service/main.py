from io import BytesIO
import traceback
from scripts.VideoReport import generate_video_pdf_report
from scripts.CropFaces import *
from scripts.DetectDeepfake import *
from scripts.DetectArt import *
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
from PIL import Image
import c2pa
import tempfile
import os
import json
import cv2
import glob
import shutil
from datetime import timedelta, datetime
import time
import platform
from scripts.SearchImage import AgenticReport
from scripts.XAI.explanation.visualize import generate_xai_visualizations
from scripts.HuggingFake import query
from flask import send_file

# Import LipSync related modules
from scripts.LipSync.model import LIPINC_model
from scripts.LipSync.utils import get_color_structure_frames

# Define Video_Short exception class (this should ideally be imported from LipSync module)
class Video_Short(Exception):
    """Exception raised when video is too short for LipSync analysis."""
    pass

#Binoculars code
from Binoculars.binoculars import Binoculars
from cgitb import text
import json
import concurrent.futures
import requests


def process_data(json_str: str):
    # Load the JSON data
    data = json.loads(json_str)

    # Extract the required fields
    username = data.get("username", "")
    bio = data.get("bio", "")

    # Initialize lists to hold post captions and comments
    post_captions = []
    comments = []

    # Process each recent post
    for post in data.get("recent_posts", []):
        # Extract the post caption if available
        caption = post.get("caption")
        if caption:
            post_captions.append(caption)
        
        # Extract all comments from the post
        for comment in post.get("comments", []):
            comment_text = comment.get("text")
            if comment_text:
                comments.append(comment_text)

    # Prepare the final output dictionary
    final_output = {
        "username": username,
        "bio": bio,
        "post_captions": post_captions,
        "comments": comments
    }


    # Create a flattened string representation for evaluation
    text_for_evaluation = (
        f"Username: {username}\n"
        f"Bio: {bio}\n"
        f"Post Captions: {'; '.join(post_captions)}\n"
        f"Comments: {'; '.join(comments)}"
    )
    return text_for_evaluation

# Example usage of evaluate_text function from Binoculars
def evaluate_text(mode, text):
    """
    Evaluates the given text using the Binoculars tool.
    
    Parameters:
        mode (str): The mode to use for evaluation, either "accuracy" or "low-fpr".
        text (str): The text string to be checked.
        
    Returns:
        tuple: A tuple containing the computed score and the prediction.
    """
    bino = Binoculars(mode=mode)
    score = bino.compute_score(text)
    prediction = bino.predict(text)
    return score, prediction
    
# Function to get descriptive result from LipSync analysis
def get_result_description(real_p):
    if real_p <= 1 and real_p >= 0.99:
        return 'This sample is certainly real.'
    elif real_p < 0.99 and real_p >= 0.75:
        return 'This sample is likely real.'
    elif real_p < 0.75 and real_p >= 0.25:
        return 'This sample is maybe real.'
    elif real_p < 0.25 and real_p >= 0.01:
        return 'This sample is unlikely real.'
    elif real_p < 0.01 and real_p >= 0:
        return 'There is no chance that the sample is real.'
    else:
        return 'Error'

def getc2pa(image):
    try:
        # Create a reader from a file path
        reader = c2pa.Reader.from_file(image)
        return reader.json()
    except Exception as err:
        return {"error": str(err)}


def convert_blob_to_image(blob):
    """Converts a blob to an image and saves it."""
    # Convert blob to bytes-like object
    byte_stream = io.BytesIO(blob)
    # Open image using Pillow
    image = Image.open(byte_stream)
    # Save the image
    image.save("output_image.jpg")


def extract_frames(video_path, output_dir="frames", interval=1):
    """
    Extract frames from a video file at specified intervals
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Base directory to save frames
        interval (int): Interval in seconds between frames
    
    Returns:
        tuple: (Path to the created output directory, video FPS, total frames extracted)
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create unique output directory
    base_dir = output_dir
    existing_dirs = glob.glob(f"{base_dir}_*")
    next_num = 1
    
    if existing_dirs:
        # Extract numbers from existing directories
        dir_nums = [int(dir.split('_')[-1]) for dir in existing_dirs if dir.split('_')[-1].isdigit()]
        if dir_nums:
            next_num = max(dir_nums) + 1
    
    output_dir = f"{base_dir}_{next_num}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step based on interval
    frame_step = int(fps * interval)
    
    frame_count = 0
    saved_count = 0
    
    # Read and save frames
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_step == 0:
            saved_count += 1
            output_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
        
        frame_count += 1
    
    video.release()
    # Ensure the video is closed before returning
    del video
    
    # Give the OS a moment to fully release the file handle (especially important on Windows)
    
    return output_dir, fps, saved_count


def safe_delete_file(file_path, max_attempts=5, delay=1.0):
    """
    Safely delete a file with multiple attempts, handling Windows file lock issues
    
    Args:
        file_path (str): Path to the file to delete
        max_attempts (int): Maximum number of deletion attempts
        delay (float): Delay between attempts in seconds
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    if not os.path.exists(file_path):
        return True
        
    for attempt in range(max_attempts):
        try:
            os.unlink(file_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                continue
            else:
                # If we can't delete it after max attempts, just report it but don't fail
                print(f"Warning: Could not delete temporary file {file_path}")
                return False


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def handle_root():
    return 'backend api'


@app.route('/extension.crx')
def send_report():
    return send_file('fence_ai.crx')


@app.route('/detect_image', methods=['POST'])
def handle_detect_image():
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename if file.filename else ''
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        file.save(temp_file.name)
        manifest = getc2pa(temp_file.name)
        img = cv2.imread(temp_file.name)
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
            # Get model score 
            model_score = query(temp_file.name)
            print(f"Model raw score: {model_score}")

            # Process manifest
            if isinstance(manifest, str):
                try:
                    manifest_data = json.loads(manifest)
                except json.JSONDecodeError:
                    manifest_data = manifest
            else:
                manifest_data = manifest
            manifest_str = json.dumps(manifest) if not isinstance(manifest, str) else manifest

            # Generate visualizations and get report
            segmented = generate_xai_visualizations(temp_file.name, r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/XAI/model/checkpoint/ff_attribution.ckpt")
            report = AgenticReport(temp_file.name, "fake", manifest_str, output=model_score)
            print(f"Report: {report}")
           
            llm_score = report.get('llm_score', 0.5) # Default to 0.5 if not found
            print(f"LLM analysis score: {llm_score}")
            print(llm_score)
            print(model_score)
            # Calculate combined score
            combined_score = (0.4 * llm_score) + (0.6 * model_score[1]['score'])
            print(f"Combined score: {combined_score}")

            # Clean up
            safe_delete_file(temp_file.name)

            # Return response with scores
            print(model_score)
            return {
                # 'model_score': float(model_score[1]['score']),
                # 'llm_score': float(llm_score), 
                # 'combined_score': float(combined_score),
                'deepfake': model_score, #[1]['score'],
                'manifest': manifest_data,
                'report': report,
                'segmented': segmented
            }
        else:
            safe_delete_file(temp_file.name)
            return f'Unsupported file format: {file_extension}'
    else:
        return 'No image data received'

@app.route('/detect_video', methods=['POST'])
def handle_detect_video():
    temp_video = None
    frames_dir = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No video file received'}), 400
        
        file = request.files['file']
        filename = file.filename if file.filename else ''
        file_extension = filename.split('.')[-1].lower()
        
        if file_extension not in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            return jsonify({'error': f'Unsupported file format: {file_extension}'}), 400
        
        # Save video to temp file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        temp_video_path = temp_video.name
        temp_video.close()
        file.save(temp_video_path)
        
        # --- Lip-Sync Analysis ---
        lip_sync_result = None
        try:
            # Extract lip features
            length_error, _, combined_frames, residue_frames, _, _ = get_color_structure_frames(
                5, temp_video_path  # n_frames=5
            )
            
            if not length_error:
                # Reshape for model input
                combined_frames = np.reshape(combined_frames, (1,) + combined_frames.shape)
                residue_frames = np.reshape(residue_frames, (1,) + residue_frames.shape)
                
                # Predict
                prediction = lip_sync_model.predict([combined_frames, residue_frames])
                real_prob = round(float(prediction[0][1]), 3)
                lip_sync_result = {
                    'real_probability': real_prob,
                    'fake_probability': round(1 - real_prob, 3),
                    'is_fake': real_prob < 0.5  # Adjust threshold as needed
                }
        except Exception as e:
            print(f"Lip-sync analysis failed: {str(e)}")
        
        # --- Existing Frame-by-Frame Analysis ---
        interval = int(request.form.get('interval', 1))
        frames_dir, fps, frame_count = extract_frames(temp_video_path, interval=interval)
        
        results = []
        fake_timestamps = []
        
        # Load the LipSync model
        lip_sync_model = LIPINC_model()
        checkpoint_path = "scripts/LipSync/checkpoints/FakeAv.hdf5"
        lip_sync_model.load_weights(checkpoint_path)
        
        for i in range(1, frame_count + 1):
            frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
            
            # Get C2PA manifest
            manifest = getc2pa(frame_path)
            
            # Load and analyze the image
            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb)
            
            predictions = analyze_image(face_pil)
            fusion = predictions.get('fusion')
            
            # Calculate timestamp for this frame
            timestamp_seconds = (i - 1) * interval
            timestamp = str(timedelta(seconds=timestamp_seconds))
            
            # Determine if frame is fake
            is_fake = False
            if fusion is not None and (fusion > 0 or (-0.000001 <= fusion <= -0.23)):
                is_fake = True
                fake_timestamps.append({
                    'timestamp': timestamp,
                    'frame_number': i,
                    'fusion_score': fusion
                })
            
            # Add result for this frame
            results.append({
                'frame_number': i,
                'timestamp': timestamp,
                'prediction': "Fake" if is_fake else "Real",
                'fusion_score': fusion,
                'is_fake': is_fake
            })
        
        # LipSync processing
        n_frames = 5  # number of local frames
        length_error, face, combined_frames, residue_frames, l_id, g_id = get_color_structure_frames(n_frames, temp_video_path)
        if length_error:
            raise Video_Short()
        
        combined_frames = np.reshape(combined_frames, (1,) + combined_frames.shape)
        residue_frames = np.reshape(residue_frames, (1,) + residue_frames.shape)
        
        lip_sync_result = lip_sync_model.predict([combined_frames, residue_frames])
        lip_sync_result = round(float(lip_sync_result[0][1]), 3)
        
        # Return the analysis results
        response_data = {
            'video_analysis': {
                'total_frames_analyzed': frame_count,
                'frame_interval': interval,
                'fps': fps,
                'duration': str(timedelta(seconds=frame_count * interval)),
                'results': results,
                'fake_frames_detected': len(fake_timestamps),
                'fake_timestamps': fake_timestamps
            },
            'lip_sync_analysis': lip_sync_result if lip_sync_result else "Analysis failed"
        }
        
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
        
        # Safely delete the temp video file
        if temp_video_path and os.path.exists(temp_video_path):
            safe_delete_file(temp_video_path)
        return jsonify(response_data)
        
    except Exception as e:
        # Clean up on error
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
            
        if temp_video and hasattr(temp_video, 'name') and os.path.exists(temp_video.name):
            safe_delete_file(temp_video.name)
        
        return jsonify({'error': str(e)}), 500

from scripts.Report import generate_pdf_report, generate_case_number
import os
import time

# Create directory for reports if it doesn't exist
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/generate_report', methods=['POST', 'OPTIONS'])
def handle_generate_report():
    try:
        # Handle OPTIONS request for CORS
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            return response
            
        print(f"Received report generation request")
        
        # Get input data with better error handling
        data = None
        if request.is_json:
            data = request.json
            print(f"Received JSON data with keys: {list(data.keys()) if data else 'None'}")
        else:
            print(f"Request is not JSON. Content type: {request.content_type}")
            data = request.form.to_dict()
            print(f"Extracted form data: {list(data.keys()) if data else 'None'}")
            
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Required fields
        investigator_name = data.get('investigator_name', 'Guard.ai Analyst')
        case_number = data.get('case_number') or generate_case_number()
        
        print(f"Processing report request for case #{case_number}")
        print(f"Investigator: {investigator_name}")
        
        # Check if we have an image file or path
        image_path = None
        temp_image = None
        
        if 'image_path' in data:
            image_path = data.get('image_path')
            print(f"Using provided image path: {image_path}")
        elif 'file' in request.files:
            # Save the uploaded file
            file = request.files['file']
            filename = file.filename if file.filename else f"case_{case_number}.jpg"
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            file.save(temp_image.name)
            image_path = temp_image.name
            print(f"Saved uploaded file to: {image_path}")
        
        # Get analysis results
        analysis_results = data.get('analysis_results')
        if not analysis_results and 'results' in data:
            analysis_results = data.get('results')
            
        # If analysis_results is a string, parse it to JSON
        if isinstance(analysis_results, str):
            try:
                analysis_results = json.loads(analysis_results)
                print(f"Parsed analysis_results from JSON string")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse analysis_results as JSON, keeping as string")
                # Keep it as a string if parsing fails
            
        # Extract grad_cam_path if available
        grad_cam_path = data.get('grad_cam_path')
        
        # Debug log the analysis_results structure
        print(f"Analysis results type: {type(analysis_results).__name__}")
        if isinstance(analysis_results, dict):
            print(f"Analysis results keys: {list(analysis_results.keys())}")
            
        # Validate required inputs
        if not image_path:
            return jsonify({'error': 'No image provided (either as path or file upload)'}), 400
        if not analysis_results:
            return jsonify({'error': 'No analysis results provided'}), 400
            
        # Check if image exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file does not exist at path: {image_path}'}), 400
        
        # Generate report PDF
        timestamp = int(time.time())
        pdf_filename = f"DeepfakeReport_{case_number}_{timestamp}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        # Print debug info
        print(f"Generating report to: {pdf_path}")
        print(f"Input image: {image_path}")
        
        # Call the report generator
        generate_pdf_report(
            output_path=pdf_path,
            case_number=case_number,
            investigator_name=investigator_name,
            analysis_results=analysis_results,
            image_path=image_path,
            grad_cam_path=grad_cam_path
        )
        
        # Clean up temporary file if we created one
        if temp_image and os.path.exists(temp_image.name):
            try:
                os.unlink(temp_image.name)
                print(f"Cleaned up temporary file: {temp_image.name}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
        
        # Return the PDF file
        print(f"Sending file: {pdf_path}")
        return send_file(
            pdf_path, 
            as_attachment=True, 
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error generating report: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return jsonify({
            'error': str(e), 
            'traceback': traceback_str,
            'status': 'failed'
        }), 500

@app.route('/generate_video_report', methods=['POST', 'OPTIONS'])
def handle_generate_video_report():
    try:
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            return response
            
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        # Extract required information
        video_analysis = data.get("analysis_results", {})
        investigator_name = data.get("investigator_name", "AI Detection System")
        case_number = data.get("case_number", generate_case_number())
        
        # Print debug information
        print(f"Generating video report for case #{case_number}")
        print(f"Investigator: {investigator_name}")
        print(f"Analysis keys: {list(video_analysis.keys()) if isinstance(video_analysis, dict) else 'Not a dict'}")
        
        # Create a unique filename with timestamp for the report
        timestamp = int(time.time())
        filename = f"video_report_{case_number}_{timestamp}.pdf"
        report_path = os.path.join(REPORTS_DIR, filename)
        
        # Extract sample frames if provided
        sample_frames = data.get("sample_frames", [])
        video_path = data.get("video_path", None)
        
        # Get model path for XAI visualizations
        model_path = data.get("model_path", r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/XAI/model/checkpoint/ff_attribution.ckpt")
        
        print(f"Starting video report generation: {report_path}")
        print(f"Using model path: {model_path}")
        print(f"Sample frames count: {len(sample_frames)}")
        
        # Ensure the lip sync analysis is included in the video_analysis if available
        if "lip_sync_analysis" not in video_analysis and "lip_sync_analysis" in data:
            video_analysis["lip_sync_analysis"] = data["lip_sync_analysis"]
        
        # Generate the report
        generate_video_pdf_report(
            report_path, 
            case_number, 
            investigator_name, 
            video_analysis,
            video_path,
            sample_frames,
            model_path
        )
        
        print(f"Video report successfully generated at: {report_path}")
        
        # Return the PDF report file
        return send_file(
            report_path, 
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error generating video report: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return jsonify({
            "error": str(e), 
            "traceback": traceback_str,
            "status": "failed"
        }), 500



@app.route('/evaluate_profile', methods=['POST'])
def evaluate_profile():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON input provided"}), 400

        # --- Text Evaluation ---
        json_str = json.dumps(data)
        text_for_eval = process_data(json_str)
        text_score, text_prediction = evaluate_text("accuracy", text_for_eval)

        # --- Image Evaluation with Deepfake, GradCAM, and LIME ---
        image_urls = []
        if data.get("profile_picture_url"):
            image_urls.append(data["profile_picture_url"])
        if "recent_posts" in data:
            for post in data["recent_posts"]:
                if post.get("image_url"):
                    image_urls.append(post["image_url"])

        # Define a new evaluation function that runs three models in parallel
        def evaluate_image(url):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    return {"url": url, "error": f"Failed to download image; status code {response.status_code}"}
                # Save downloaded image to temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp_file.write(response.content)
                tmp_file.close()
                # Run deepfake, gradcam, and lime evaluations concurrently
                with concurrent.futures.ThreadPoolExecutor() as inner_executor:
                    future_deepfake = inner_executor.submit(query, tmp_file.name)
                    # Adjust the model/checkpoint paths as required
                    future_gradcam = inner_executor.submit(generate_xai_visualizations, tmp_file.name, r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/XAI/model/checkpoint/ff_attribution.ckpt")
                    future_lime = inner_executor.submit(generate_lime_visualizations, tmp_file.name, r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/XAI/model/checkpoint/lime_model.ckpt")
                    deepfake_result = future_deepfake.result()
                    gradcam_result = future_gradcam.result()
                    lime_result = future_lime.result()
                safe_delete_file(tmp_file.name)
                return {
                    "url": url,
                    "deepfake": deepfake_result,
                    "gradcam": gradcam_result,
                    "lime": lime_result
                }
            except Exception as e:
                return {"url": url, "error": str(e)}

        image_eval_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_image, url): url for url in image_urls}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                image_eval_results.append(result)

        # Extract deepfake scores from each image evaluation (assuming the structure is similar to your /detect_image endpoint)
        scores = []
        for entry in image_eval_results:
            try:
                # Here we assume the deepfake result follows the format: result[1]['score']
                score = entry["deepfake"][1]["score"]
                scores.append(score)
            except Exception:
                pass
        avg_image_score = sum(scores) / len(scores) if scores else 0.5

        # --- Final Overall Evaluation ---
        # Combine text and image evaluation scores (40% text, 60% image deepfake score)
        final_score = (0.4 * text_score) + (0.6 * avg_image_score)
        final_verdict = "Real" if final_score >= 0.5 else "Fake"

        final_report = {
            "text_evaluation": {
                "score": text_score,
                "prediction": text_prediction,
                "text": text_for_eval
            },
            "image_evaluation": {
                "average_deepfake_score": avg_image_score,
                "individual_results": image_eval_results
            },
            "final_report": {
                "final_score": final_score,
                "verdict": final_verdict
            }
        }

        # --- Generate a Detailed Overall PDF Report on the Profile ---
        # Determine investigator name (if provided) and generate a case number
        investigator_name = data.get("investigator_name", "Guard.ai Analyst")
        case_number = data.get("case_number") or generate_case_number()
        timestamp = int(time.time())
        pdf_filename = f"profile_report_{case_number}_{timestamp}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

        # For the PDF report, we can use the profile picture if available.
        profile_image_path = None
        if data.get("profile_picture_url"):
            try:
                response = requests.get(data["profile_picture_url"], timeout=10)
                if response.status_code == 200:
                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    tmp_img.write(response.content)
                    tmp_img.close()
                    profile_image_path = tmp_img.name
                else:
                    print(f"Warning: Failed to download profile image; status code {response.status_code}")
            except Exception as e:
                print(f"Warning: Exception downloading profile image: {str(e)}")

        # Call the report generator (assumes generate_pdf_report can accept a detailed analysis_results dict)
        generate_pdf_report(
            output_path=pdf_path,
            case_number=case_number,
            investigator_name=investigator_name,
            analysis_results=final_report,
            image_path=profile_image_path,
            grad_cam_path=""  # Optionally, you could pass a representative gradcam image path if available
        )

        # Clean up temporary profile image file if downloaded
        if profile_image_path and os.path.exists(profile_image_path):
            safe_delete_file(profile_image_path)

        # Return the final JSON report with a link/path to the PDF report
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
)

    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback_str
        }), 500

if __name__ == '__main__':
    app.run()
lip_sync_model = LIPINC_model()
lip_sync_model.load_weights(r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/LipSync/checkpoints/FakeAv.hdf5")
