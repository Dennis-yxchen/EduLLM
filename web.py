import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'web', 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'web', 'static')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx'}

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOADS_DIR'] = UPLOADS_DIR
app.secret_key = 'secret_key_here'  # Change to something secure

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_folders_and_files():
    """
    Counts how many folders (knowledge bases) exist in the uploads folder,
    and how many total files are in all subfolders.
    Returns (num_folders, num_files).
    """
    if not os.path.exists(UPLOADS_DIR):
        return 0, 0

    folder_count = 0
    file_count = 0
    for item in os.listdir(UPLOADS_DIR):
        subpath = os.path.join(UPLOADS_DIR, item)
        if os.path.isdir(subpath):
            folder_count += 1
            # Count files in this folder
            files_in_folder = os.listdir(subpath)
            file_count += len(files_in_folder)
    return folder_count, file_count

@app.route('/', methods=['GET'])
def index():
    """
    Main page:
      - No upload area here.
      - Shows the number of knowledge bases (folders under uploads/)
      - Shows the total file count across all knowledge bases.
    """
    kb_count, total_files = count_folders_and_files()
    return render_template('index.html', kb_count=kb_count, total_files=total_files)

@app.route('/create_kb', methods=['POST'])
def create_kb():
    """
    Create a new knowledge base folder under /uploads.
    Expects a 'kb_name' field from the form or AJAX.
    """
    kb_name = request.form.get('kb_name', '').strip()
    if not kb_name:
        flash("No Knowledge Base name provided.")
        return redirect(url_for('index'))

    safe_folder_name = secure_filename(kb_name)
    kb_path = os.path.join(UPLOADS_DIR, safe_folder_name)
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        flash(f"Knowledge Base '{safe_folder_name}' created successfully!")
    else:
        flash(f"Knowledge Base '{safe_folder_name}' already exists!")

    return redirect(url_for('index'))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """
    Upload a file into a specific knowledge base folder.
    Expects 'kb_folder' to identify which folder to store the file in.
    """
    kb_folder = request.form.get('kb_folder', '')
    if not kb_folder:
        flash("No knowledge base folder specified.")
        return redirect(url_for('index'))

    kb_path = os.path.join(UPLOADS_DIR, kb_folder)
    if not os.path.exists(kb_path):
        flash(f"Knowledge Base '{kb_folder}' does not exist.")
        return redirect(url_for('index'))

    # Check if file part is in the request
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(kb_path, filename))
        flash(f"File '{filename}' uploaded to Knowledge Base '{kb_folder}'!")
    else:
        flash("File type not allowed.")

    return redirect(url_for('index'))

@app.route('/kb_list', methods=['GET'])
def kb_list():
    """
    Return a JSON list of knowledge base folders for use in the modal's dropdown or UI.
    """
    if not os.path.exists(UPLOADS_DIR):
        return jsonify([])

    dirs = []
    for item in os.listdir(UPLOADS_DIR):
        subpath = os.path.join(UPLOADS_DIR, item)
        if os.path.isdir(subpath):
            dirs.append(item)
    return jsonify(dirs)

@app.route('/uploads/<kb_folder>/<filename>')
def serve_file(kb_folder, filename):
    """
    Serve a file from within a knowledge base folder.
    """
    safe_kb_folder = secure_filename(kb_folder)
    safe_filename = secure_filename(filename)
    file_path = os.path.join(UPLOADS_DIR, safe_kb_folder, safe_filename)
    if not os.path.exists(file_path):
        flash("File does not exist.")
        return redirect(url_for('index'))
    return send_from_directory(os.path.join(UPLOADS_DIR, safe_kb_folder), safe_filename)

# Optional: a route to edit or list files in a specific KB could be added
# but for brevity we focus on main functionality now.

if __name__ == '__main__':
    os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)
    app.run(debug=True, port=5001)
